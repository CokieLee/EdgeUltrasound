#!/usr/bin/env python3
"""Load int16 parameters, infer one event, check score and delayed input replay."""
import argparse
import struct
import time
from pathlib import Path
from cnn_reference import load_model,pack_weights,load_input,infer


def read_exact(port,count,timeout):
    data=bytearray(); end=time.monotonic()+timeout
    while len(data)<count:
        if time.monotonic()>end:
            raise TimeoutError(f'Expected {count} response bytes; got {len(data)}')
        data.extend(port.read(count-len(data)))
    return bytes(data)


def send_all(port,data):
    offset=0
    while offset<len(data):
        n=port.write(data[offset:])
        if not n: raise TimeoutError('Serial write stalled')
        offset+=n


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port',required=True)
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--event',type=int,default=0)
    p.add_argument('--timeout',type=float,default=60)
    args=p.parse_args()
    params,cfg=load_model(args.weights,args.config)
    x=load_input(args.input,args.event)
    expected=infer(params,cfg,x)
    config_bytes=bytes(v for pair in zip(cfg['right_shifts'],cfg['bias_left_shifts']) for v in pair)
    weights=pack_weights(params).tobytes()
    payload=x.tobytes(order='C')
    print(f'Event {args.event}: {x.shape}; {len(payload)} input bytes; reference score {expected}')
    print('This engine consumes one selected event. Event 0 matches channel_reduction=1.')
    import serial
    with serial.Serial(args.port,115200,timeout=0.05,write_timeout=args.timeout,
                       xonxoff=False,rtscts=False,dsrdtr=False) as port:
        port.reset_input_buffer()
        send_all(port,b'W'+config_bytes+weights)
        if read_exact(port,1,args.timeout)!=b'\xa1':
            raise RuntimeError('Weight-load acknowledgement invalid; reset FPGA')
        print('Weights loaded; sending input and waiting for inference.')
        t=time.monotonic()
        send_all(port,b'I'+payload)
        if read_exact(port,1,args.timeout)!=b'\xd1':
            raise RuntimeError('Inference response marker invalid; reset FPGA')
        actual=struct.unpack('<h',read_exact(port,2,args.timeout))[0]
        replay=read_exact(port,len(payload),args.timeout)
        if read_exact(port,1,args.timeout)!=b'\xd2':
            raise RuntimeError('Final marker invalid; reset FPGA')
        if replay!=payload:
            raise RuntimeError('Returned original input differs from sent data')
        if actual!=expected:
            raise RuntimeError(f'FPGA score {actual} differs from reference {expected}')
        print(f'PASS: score {actual} matches integer reference; all {len(payload)} original input bytes match.')
        print(f'Input upload + inference + input replay: {time.monotonic()-t:.3f} s (not pure inference latency).')


if __name__=='__main__':
    main()
