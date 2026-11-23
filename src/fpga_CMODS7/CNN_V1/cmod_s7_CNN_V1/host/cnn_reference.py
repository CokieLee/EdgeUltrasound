"""Bit-exact integer reference and file contract for cnn_engine.sv."""
import json
from pathlib import Path
import numpy as np

LAYERS = [('stem', (8,1,1,1)), ('conv1',(16,8,3,3)),
          ('conv2',(32,16,3,3)), ('conv3',(32,32,3,3)), ('head',(1,32))]


def load_model(path, config_path):
    params = {}
    with np.load(path, allow_pickle=False) as archive:
        for name, shape in LAYERS:
            for suffix, expected in [('weight',shape), ('bias',(shape[0],))]:
                key = f'{name}.{suffix}'
                value = archive[key]
                if value.shape != expected or value.dtype.kind!='i' or value.dtype.itemsize!=2:
                    raise ValueError(f'{key}: expected signed int16 {expected}, got {value.dtype} {value.shape}')
                params[key] = value.astype('<i2')
    cfg = json.loads(Path(config_path).read_text())
    for key in ('right_shifts','bias_left_shifts'):
        values = cfg[key]
        if len(values)!=5 or any(type(v) is not int or not 0<=v<=31 for v in values):
            raise ValueError(f'{key} requires five integers in 0..31')
    return params, cfg


def pack_weights(params):
    return np.concatenate([params[f'{name}.{suffix}'].reshape(-1)
                           for name,_ in LAYERS for suffix in ('weight','bias')]).astype('<i2')


def requantize(acc, right_shift, relu):
    value = np.right_shift(acc, right_shift)
    return np.clip(value, 0 if relu else -32768, 32767).astype(np.int16)


def conv(x, w, b, stride, pad, right_shift, bias_shift, relu):
    x = x.astype(np.int64)
    w = w.astype(np.int64)
    k = w.shape[2]
    height = (x.shape[1]+2*pad-k)//stride+1
    width = (x.shape[2]+2*pad-k)//stride+1
    padded = np.pad(x, ((0,0),(pad,pad),(pad,pad)))
    out = np.broadcast_to((b.astype(np.int64)<<bias_shift)[:,None,None],
                          (w.shape[0],height,width)).copy()
    for ky in range(k):
        for kx in range(k):
            patch = padded[:, ky:ky+height*stride:stride, kx:kx+width*stride:stride]
            out += np.einsum('oi,ihw->ohw',w[:,:,ky,kx],patch,dtype=np.int64)
    return requantize(out,right_shift,relu)


def infer(params,cfg,x,return_layers=False):
    x = np.asarray(x)
    if x.shape!=(1,32,156) or x.dtype.kind!='i' or x.dtype.itemsize!=2:
        raise ValueError('Engine input must be signed int16 with shape (1,32,156)')
    outputs = {}
    for i,(name,_) in enumerate(LAYERS[:4]):
        x = conv(x,params[name+'.weight'],params[name+'.bias'],
                 1 if i==0 else 2,0 if i==0 else 1,
                 cfg['right_shifts'][i],cfg['bias_left_shifts'][i],i!=0)
        outputs[name] = x
    pooled = (x.astype(np.int64).sum(axis=(1,2))//80).astype(np.int16)
    outputs['pool'] = pooled
    acc = (params['head.weight'].astype(np.int64) @ pooled.astype(np.int64))
    acc += params['head.bias'].astype(np.int64)<<cfg['bias_left_shifts'][4]
    score = int(requantize(acc,cfg['right_shifts'][4],False)[0])
    return (score,outputs) if return_layers else score


def load_input(path,event):
    x = np.load(path,allow_pickle=False)
    if x.dtype.kind!='i' or x.dtype.itemsize!=2:
        raise ValueError('Input must already be quantized signed int16; no implicit conversion')
    if x.ndim==4:
        if x.shape[0]!=1:
            raise ValueError('Only batch size 1 is supported')
        x=x[0]
    if x.ndim!=3 or x.shape[1:]!=(32,156) or not 0<=event<x.shape[0]:
        raise ValueError('Expected [1,events,32,156] or [events,32,156]')
    return np.ascontiguousarray(x[event:event+1],dtype='<i2')


def make_demo(folder):
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(7)
    params={}
    for name,shape in LAYERS:
        params[name+'.weight']=rng.integers(-2,3,size=shape,dtype=np.int16)
        params[name+'.bias']=rng.integers(-4,5,size=(shape[0],),dtype=np.int16)
    # Make the untrained smoke-test head produce a nonzero score.
    params['head.bias'][:]=123
    cfg={'right_shifts':[0,2,2,2,0], 'bias_left_shifts':[0]*5,
         'description':'UNTRAINED synthetic smoke test; not a rat classifier'}
    np.savez(folder/'demo_weights.npz',**params)
    (folder/'demo_config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    np.save(folder/'demo_input.npy',rng.integers(-16,17,size=(1,1,32,156),dtype=np.int16))
    return params,cfg


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--make-demo',type=Path,required=True)
    args=p.parse_args()
    params,cfg=make_demo(args.make_demo)
    print('UNTRAINED demo score:',infer(params,cfg,load_input(args.make_demo/'demo_input.npy',0)))
