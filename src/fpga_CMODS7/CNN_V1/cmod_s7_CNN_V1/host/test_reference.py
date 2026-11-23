import unittest
import tempfile
from pathlib import Path
import numpy as np
from cnn_reference import conv,requantize,make_demo,load_model,pack_weights,infer,load_input

class ReferenceTests(unittest.TestCase):
    def test_convolution_against_independent_scalar_loops(self):
        rng=np.random.default_rng(5)
        x=rng.integers(-20,21,size=(2,4,5),dtype=np.int16)
        w=rng.integers(-4,5,size=(3,2,3,3),dtype=np.int16)
        b=np.array([-3,2,1],dtype=np.int16)
        actual=conv(x,w,b,2,1,2,1,True)
        expected=np.zeros((3,2,3),dtype=np.int16)
        for o in range(3):
            for y in range(2):
                for xx in range(3):
                    a=int(b[o])*2
                    for i in range(2):
                        for ky in range(3):
                            for kx in range(3):
                                iy=y*2+ky-1; ix=xx*2+kx-1
                                if 0<=iy<4 and 0<=ix<5:
                                    a+=int(x[i,iy,ix])*int(w[o,i,ky,kx])
                    expected[o,y,xx]=max(0,min(32767,a>>2))
        np.testing.assert_array_equal(actual,expected)

    def test_signed_rounding_saturation_relu(self):
        x=np.array([-100000,-3,-1,0,3,100000],dtype=np.int64)
        np.testing.assert_array_equal(requantize(x,1,False),[-32768,-2,-1,0,1,32767])
        np.testing.assert_array_equal(requantize(x,1,True),[0,0,0,0,1,32767])

    def test_complete_network_and_parameter_contract(self):
        with tempfile.TemporaryDirectory() as t:
            params,cfg=make_demo(t)
            p,c=load_model(Path(t)/'demo_weights.npz',Path(t)/'demo_config.json')
            x=load_input(Path(t)/'demo_input.npy',0)
            score,out=infer(p,c,x,True)
            self.assertEqual(pack_weights(p).size,15105)
            self.assertEqual(out['stem'].shape,(8,32,156))
            self.assertEqual(out['conv1'].shape,(16,16,78))
            self.assertEqual(out['conv2'].shape,(32,8,39))
            self.assertEqual(out['conv3'].shape,(32,4,20))
            self.assertEqual(out['pool'].shape,(32,))
            self.assertTrue(-32768<=score<=32767)

if __name__=='__main__': unittest.main()
