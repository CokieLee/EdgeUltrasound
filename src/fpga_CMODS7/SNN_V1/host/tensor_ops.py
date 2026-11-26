"""Tensor shapes, packing, and fixed-point convolution for the FPGA."""
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

