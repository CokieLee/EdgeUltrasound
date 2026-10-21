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
