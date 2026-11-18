from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'host'))
from cnn_reference import make_demo,pack_weights,load_input,infer

root=Path(__file__).resolve().parents[1]
folder=root/'build'/'vectors'; folder.mkdir(parents=True,exist_ok=True)
params,cfg=make_demo(folder)
x=load_input(folder/'demo_input.npy',0)
score,layers=infer(params,cfg,x,True)

def save(name,array):
    values=np.asarray(array).reshape(-1)
    (folder/(name+'.mem')).write_text(''.join(f'{int(v)&65535:04x}\n' for v in values))

save('weights',pack_weights(params)); save('input',x); save('score',[score])
for name,value in layers.items(): save(name,value)
print('Generated independent integer reference vectors; expected score',score)
