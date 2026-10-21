# EdgeUltrasound



## Results for MidTerm Report
SoS regression MSE: 0.0021612638 (fat fraction)^2
SoS uses stored predictions from a regression fitted on all rats.

CNN
Parameters: 0.2529 MiB
Buffers: 0.0000 MiB
Total model size: 0.2529 MiB
SNN
Parameters: 0.2529 MiB
Buffers: 0.0001 MiB
Total model size: 0.2529 MiB

Network MSE in logged units; sample stdev across runs (ddof=1):
CNN: 17 v2 runs, 25 epochs
  Final epoch: mean=0.0019377276, stdev=0.0011297484
  Best epoch:  mean=0.0013047348, stdev=0.0008201375
SNN: 12 v2 runs, 75 epochs
  Final epoch: mean=0.0033036875, stdev=0.0049808150
  Best epoch:  mean=0.0024713317, stdev=0.0044423478
Target normalization is absent from logs. If targets were /100, use the fraction-scale SoS MSE.


Inference timing

CNN | CPU (cpu) | shape=[1, 384, 64, 312] | 1 file(s)
  Experiment: v2; init_params={'in_channels': 384, 'spiking': False}
  Metric                             N              Mean      Sample stdev
  Latency (logged ms)              250     0.00760004751     0.00127407008

SNN | CPU (cpu) | shape=[1, 384, 64, 312] | 1 file(s)
  Experiment: v2; init_params={'in_channels': 384, 'spiking': True}
  Metric                             N              Mean      Sample stdev
  Latency (logged ms)              250      0.0188438164     0.00216305517

CNN | A100 (cuda) | shape=[1, 384, 64, 312] | 1 file(s)
  Experiment: v2; init_params={'in_channels': 384, 'spiking': False}
  Metric                             N              Mean      Sample stdev
  Latency (logged ms)              250     0.00103633054    7.04141213e-05

SNN | A100 (cuda) | shape=[1, 384, 64, 312] | 1 file(s)
  Experiment: v2; init_params={'in_channels': 384, 'spiking': True}
  Metric                             N              Mean      Sample stdev
  Latency (logged ms)              250     0.00829910348    0.000236295804

CNN | T4 GPU (cuda) | shape=[1, 384, 64, 312] | 1 file(s)
  Experiment: v2; init_params={'in_channels': 384, 'spiking': False}
  Metric                             N              Mean      Sample stdev
  Latency (logged ms)              250    0.000921416576    7.64142658e-05

SNN | T4 GPU (cuda) | shape=[1, 384, 64, 312] | 1 file(s)
  Experiment: v2; init_params={'in_channels': 384, 'spiking': True}
  Metric                             N              Mean      Sample stdev
  Latency (logged ms)              250      0.0077425506     0.00513335261

CNN | v5e1 TPU (tpu) | shape=[1, 384, 64, 312] | 1 file(s)
  Experiment: v2; init_params={'in_channels': 384, 'spiking': False}
  Metric                             N              Mean      Sample stdev
  Latency (logged ms)              250     0.00056928038     0.00012183885

SNN | v5e1 TPU (tpu) | shape=[1, 384, 64, 312] | 1 file(s)
  Experiment: v2; init_params={'in_channels': 384, 'spiking': True}
  Metric                             N              Mean      Sample stdev
  Latency (logged ms)              250     0.00828938972     0.00126810411