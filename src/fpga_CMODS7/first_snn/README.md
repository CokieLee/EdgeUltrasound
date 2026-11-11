# Design of the SNN Core:
11/10 - Ethan

# TODO: make a quick figure for Tergel
# TODO: first pass at the architecture using the BRAM, DSP slices, and neurons
you should explicitly answer specifically how you will handle loading in the weights and neurons and how we will handle various layers such as the CNN vs LIF neurons
its ok for my first pass to be crap. Tergel and AI will get me somewhere better.

### 1. High Level Problem Overview:
Our FPGA is a Spartan7 with 
- 14,600 LUTS
- 29,200 Flip-flops
- 202.5 KiB of BRAM, in 45 separate 36-Kibit BRAM blocks.
- 80 DSP slices

Each DSP Slice includes a 25×18-bit multiplier and a 48-bit accumulator
I also have 4MB SPI flash and a 12 MHz clock.

My teams original neural network was 250 KB, so we were well beyond BRAM, but now we got it down to ~32KB with int16 quantization, though we haven't done any pruning.

So the major challenge I am having is trying to decide how to design my FPGA computing architecture. I can tell you more about the neural network structure, and because it has convolution and LIF neurons, I know that at the very least I will need some kind of convolving stage, which I think I can use the DSP slices for, and some kind of LIF neuron modules. But I won't be able to just directly map the neural network onto the FPGA fabric, I think I need to make reusable hardware that I can like run 1 layer at a time and load weights and data in and out of.



### 2. New Architecture:
Using downsize_factor = 2 and the latest sparsification (1 recieve event instead of 384)
We will call each recieve event an image even though it is NOT an image - its just easier to understand the network if we think of the input like an image.

The input data is [1 × 64 × 312] x int16

| Operation | Output shape, excluding batch | Meaning |
|---|---|---|
| Event selection | `1 × 64 × 312` | Keep one receive event |
| Stem: 1×1 convolution | `8 × 64 × 312` | Produce 8 learned feature maps, lots of Multiplies and Adds |
| `conv1` → `act1` | `16 × 32 × 156` | Convolution followed by LIF, lots of Multiplies and Adds, then LIF operations |
| `conv2` → `act2` | `32 × 16 × 78` | Convolution followed by LIF, lots of summations of spikes/weights and LIF operations |
| `conv3` → `act3` | `32 × 8 × 39` | Convolution followed by LIF, lots of summations of spikes/weights and LIF operations |
| Spatial average pooling | `32 × 1 × 1` | Average 312 positions separately in each feature map |
| Flatten | `32` | Arrange those averages as a vector |
| Linear head | `1` | Learned weighted sum plus bias, lots of summation of spikes/weights |
| Time averaging | `1` | Average the four per-step scalar predictions |

This model has 15,105 parameters, which is ~30KB in int16.
It is also particularly simple:
For input $X$
1. Stem layer is simply applying 8 different weights + 8 biases onto the entire input "image"
$Y_i = a_iX + b_i\qquad i=1,\ldots,8$, Y_i is [64, 312], and the total output of stem layer is [8, 64, 312]
2. Each of the feature mapped inputs is convolved with a new set of kernels in conv1,conv2,conv3. 
Conv1: [8, 64, 312] input -> [16 x 32 x 156] output
$Z_k = c_k + \sum_{j=1}^{8}K_{k,j}*Y_j \qquad k=1,\ldots,16$
We have 128 kernels - 1 unique array for each of 8 feature mapped input, and 16 new feature maps (8*16 = 128). Note that we used stride 2 (both x and y) and zero padding of 1, so its not perfect convolution but close, and explains the loss of 2 across both input dimensions.

After convolution,  we take the [16 x 32 x 156] output and apply ReLu (CNN) or it becomes the input to an LIF neuron (SNN). This means we apply to each input $f(x) = max(0,x)$ or $LIF(x,4)$ - 4 steps of LIF neuron simulation. That means we need 79k neurons because we need to apply the neuron to every single element.

Each layer of convolution after the first layer does accumulation as inputs are binary:
$I=b+\sum_i w_i s_i,\qquad s_i\in\{0,1\}$
No general multiplier is needed for that contribution. The weight can still be signed and 16-bit, and the accumulator may need more bits.DSPs can still help with accumulation, membrane leak scaling, or the final weighted readout. But you don’t need to assign a DSP multiplier to every later-layer processing lane.


3. At the end of each conv layer we have either a Relu (CNN) or a spiking neuron (SNN)
4. We flatten the final output across time and channels into 32 features and then apply a linear layer to generate 1 final feature


This tells me there are 3 major operations in order of usage:
1. Convolution
2. LIF neuron / Relu
3. Averaging / pooling

A major limitation to these is memory - we need to load in data and weights to do convolution. But the FPGA doesn't come with a built in scheduler that assigns DSPs and neurons - I need to build a scheduler. It would look like a FSM that considers which layer I am running, the input/output position, and who recieves what. It will be clear what memory addresses to read and registers to enable when the FSM is setup, like a for loop - you set the start and end indices, the computer/scheduler takes care of the rest.
A procesing lane will be central to making this work. Avoiding BRAM round trips after convolution will be helpful.
Note: the first layer of the CNN takes in ints, and outputs binary. The first layer will involve a lot of multiplication operations, while following layers will involve more accumulation/summation. This means DSP might be very powerful for the first layer, but perhaps less powerful after layer 1. 
Note: 79k of neural operations doesn't mean we are cooked - it means 79k of neural operations on reusable hardware
Note: The BRAM is organized in 45 seperate 36KB blocks, with dual ports - two ports per memory block. That means you can theoretically do 90 simultaneous memory accesses per clock cyle - thats 90 "words" (we still need to confirm word size). Note writing uses a port too - so I can choose between doing a read and a write in 1 clock cycle, 2 reads, or 2 writes, or less. Organizing the data in these blocks by simultaenous access will help a lot. Even better - one read can feed into multiple processing lanes - that can go to multiple DSPs or Neurons. So if you can reuse memory and parralelize you can have excpetional throughput. Arranging the data and reads in a smart way will make this work.



Minor example of optimization:
If step 1 is just multipling and adding a unique weight 8 times... instead of making 8 copies of our data in memory, we can combine it with step 2. 
If step 1 is:
$Y_i = a_iX + b_i,\qquad i=1,\ldots,8$
and step 2 is:
$Z_k = \sum_{j=1}^{8}K_{k,j}*Y+c_k$
then step 1 and 2 can be combined:
$Z_k = \sum_{j=1}^{8}K_{k,j}*(a_iX + b_i)+c_k$
We can precompute these new combined kernels post training. Instead of eight feature maps feeding into kernels, we can precompute these new kernels. Alternatively, we can fuse the operations. 




11/9 - Ethan - Previous Architecture

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