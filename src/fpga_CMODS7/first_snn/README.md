# Design of the SNN Core:
11/8 - Ethan

### 1. Weight Memory Footprint 11/8
In order to put the SNN onto the FPGA, we need to consider the available resources versus the current implementation of the SNN. The CMOD S7 has:
- 14,600 LUTS
- 202.5 KiB of BRAM
- 80 DSP slices

Note that the BRAM is not instantiated using the FPGA fabric, it is dedicated, hardwired memory.

As of 11/8 our SNN/CNN is ~259 KB with float32 format. Our weights alone exceed the memory on device, let alone the space needed to store neuron internal memory (membrane potentials, refractory states, and buffers for incoming data). We are way out of the park, but this doesn't mean we need to stream everything in. 1 quantization step (float32 -> int16) drops us to 129.5 KB which is manageable. 2 steps (float32 -> int8) drops us to ~65KB, which would be fantastic. Quantization and neuron pruning is likely going to save our butts here.

Let's assume we acheive one step of quantization, and possibly achieve two with 30% of our neurons pruned. We are left with ~93KB of weights, which is totally manageable. Now we need to think about or SNN/CNN structure and whether we can actually fit the network.

### 2. SNN/CNN Architecture 11/8
Our current input is an array of $(N, E, C, F)$:
[1, 384, 64, 312] X float32

where:
- $N$ is the number of frames (which are like recordings)
- $E$ is the receive_events
- $C$ is the channels (ultrasound array channels)
- $F$ is the samples recorded per channel (time)

That is a lot of data... but lets not worry about it right now. Right now lets focus on the network because its likely the network is too massive for our FPGA. Lets see! For each layer I 

| Operation | Output shape, excluding batch | What changes |
|---|---|---|
| Input | `384 × 64 × 312` | 384 input recieve events |
| `stem`: 1×1 convolution | `16 × 64 × 312` | Compresses 384 recieve events into 16 |
| `conv1` + `act1` | `32 × 32 × 156` | Doubles channels; halves both spatial dimensions |
| `conv2` + `act2` | `64 × 16 × 78` | Doubles channels; halves both spatial dimensions |
| `conv3` + `act3` | `64 × 8 × 39` | Keeps 64 channels; halves both spatial dimensions |
| Average pooling | `64 × 1 × 1` | Averages each channel’s spatial positions |
| Linear head | `1` | Combines the 64 averages into one scalar |


If our input is a [384, 64, 312] ([recieve events, channels, samples]), then:
1. The stem layer uses learned weights to produce 16 new "receive events" from the input 384 channels. This is effectively a massive matrix multiplication step. (in more detail, we can think of each recieve event as an "image" of [channels, samples] and we are applying a unique weight to each image, then summing all 384 images and adding a bias, yielding 1 new output channel. We repeat this with 16 sets of weights to get 16 new feature mapped "events").
2. Each of the three main stages is a convolution followed by an activation. In the CNN, we use RELU (y=max(0,x)), while in the SNN we use neurons, which will take in the convolution result and output binary spikes.
3. Average pooling averages 312 inputs into 64??
4. The final linear head involves a matrix multiplication and weight addition.
5. The SNN was trained with 4 steps. That means for each input, we present it to the LIF layers 4 times and then average per step outputs into one final output. 


At a high level, we have 3 kinds of computation:
1. Matrix Multiplication (CNNs)
2. Averaging
3. LIF Neuron processing

However, its layer specific:
1. Numeric matrix multiplicaton (stem, conv1)
2. Spike weighted accumulation (matrix * spike (0,1) -> less like MatMul more like conditionals)
3. LIF Neuron processing (layers following the Convolution Layers - act1,act2,act3)
4. Averaging (average pooling)

So likely we want reusable hardware consisting of an accumulation engine, a LIF update unit, memory, and a scheduler/address generator. Those are conceptual blocks; they needn’t each have separate arithmetic. The important next distinction is what must remain stored versus what can be computed, consumed, and overwritten.




Consider one neuron, with illustrative values:
- Input current: \(I=0.4\), constant across the four steps.
- Leak coefficient: \(\beta=0.9\).
- Threshold: \(\theta=1.0\).
- Initial membrane: \(u_0=0\).

Using the default SNNTorch LIF, a single neuron is calculated as:
$u_t=0.9u_{t-1}+I_t-s_{t-1}\theta$
$s_t=
\begin{cases}
1 & u_t>\theta\\
0 & \text{otherwise}
\end{cases}
$


