# Pipeline methods

## L. Guan, D.-S. Li, J.-Y. Liang, W.-J. Wang, K.-S. Ge, and X.-C. Lu, “Advances of Pipeline Model Parallelism for Deep Learning Training: An Overview,” J. Comput. Sci. Technol., vol. 39, no. 3, pp. 567–584, May 2024, doi: 10.1007/s11390-024-3872-3.

### Challenges in PMP

- devising an effective pipeline schedule strategy that determines the concurrency and learning efficiency of pipeline training
- achieving load balance between intra-node and inter-node training,
- reducing the costs of computation, storage, and communication 

### Pipeline Schedule for PMP

#### Synchronous Pipeline Schedule

- GPipe
  - the use of microbatching to reduce the number of bubbles in its pipeline structure and improve GPU utilization.

#### Asynchronous Pipeline Schedule

##### weight stashing

![gpipe](../images/pipelines/Screenshot%202024-11-14%20at%2017.04.05.png)

always use same version of weight because sync update

![pipedream](../images/pipelines/Screenshot%202024-11-14%20at%2017.11.48.png)

requires storing one version of weights for each mini-batch that is in progress in the pipeline.

- PipeDream-2BW
![PipeDream-2BW](../images/pipelines/Screenshot%202024-11-14%20at%2017.26.30.png)
  - utilizes a technique called double-buffered weight updates (2BW).
  - With the 2BW technique, for a micro-batch that has just entered the pipeline, the latest weights are used for forward pass.
  - Meanwhile, for micro-batches already in the pipeline, 2BW employs the previously cached weights for backward propagation.
  - maintain only two versions of weights,

- WPipe
  - proposes double-grouped weight updates (2GW)
  - divides model partitions into two groups, rearranges the execution order of micro-batches in the first group, and alternatively executes the update of each group.

##### weight prediction

- SpecTrain
  - utilizes the product of the smoothed gradient and the weight version differences to predict the model weights that will be used in future pipeline time steps.
  - momentum SGD

- XPipe
  - constructs the weight prediction mechanism based on the Adam[65] optimizer

- PipeMare
  - uses **learning rate rescheduling** and **discrepancy correction** to improve the statistical efficiency of asynchronous pipeline parallelism

- AvgPipe
  - employs an elastic averaging-based framework to mitigate the bubble issue in GPipe and maintain the statistical efficiency where multiple parallel pipelines are executed and each pipeline handles a batch of data per iteration.

### Comparison

![note](../images/pipelines/Screenshot%202024-11-13%20at%2015.18.02.png)

![comp](../images/pipelines/Screenshot%202024-11-13%20at%2015.10.41.png)

#### Bubble Ratio

synchronous > asynchronous

ZB-H2 approach excepted

#### Convergence

synchronous > asynchronous

#### Weights Memory

GPipe[29], DAPPLE[48], ZB[56], AMPNet[57], XPipe[60], SpecTrain[59], and PipeMare[66] have the lowest weight storage overhead, with only one copy of weights

#### Activations Memory

- GPipe, Megatron-LM, and WPipe rank second which require each GPU to store activations because of the adopted recomputation technique.
- GEMS enjoys the lowest activations memory consumption, only requiring each GPU to store one input activations.
- Chimera, which highly depends on the pipeline depth.

#### Extra Memory

- the weight stashing techniques used in PipeDream[28], PipeDream2BW[58], and WPipe[62] incur extra memory consumption.

### Load Balance for Pipeline Training

#### Load Balance for Intra-Node Training

The partitioning strategy of pipeline parallelism is to divide the computational graph of a model into multiple consecutive layer blocks (also known as **stages**), enabling parallel execution of operations within each stage.

- dynamic programming
  - used in PipeDream[28], PipeDream-2BW[58], EffTra[69], and DAPPLE[48].
- reinforcement learning
  - Alpa[70] discovers that the hierarchical search method can effectively search for model partitioning strategies, thereby contributing to load balance for intra-node training
  - AutoPipe[71] contains a planner for automatically generating a balanced pipeline partition scheme with a heuristic partition search algorithm.
  - Unity[72]
  - vPipe

- reducing the pipeline bubbles or filling them with computations
  - Microbatching
  - The “1F1B” schedule
  - Dual-/multiple-pipeline training,
    - combine two or more pipelines to reduce the number of bubbles and thus achieve more balanced pipelined training.
  - Bubble filling
    - PipeFisher

#### Load Balance for Inter-Node Training

- the mixture of pipeline parallelism and data parallelism
  - GPipe[29], PipeDream[28], PipeDream-2BW[58], DAPPLE[48], and GEMS[54]
- combining pipeline parallelism with both data parallelism and tensor parallelism
  - DistBelief[24], Piper[75], and MegatronLM[20]

### Asynchronous Pipeline Parallelism with Effective Parameter Learning

Therefore, in future research on asynchronous pipelined training, we forecast that **dynamically predicting weights based on the used optimizer** is a promising way to improve the robustness of weight prediction and enhance the training efficiency of asynchronous pipeline parallelism approaches.

### Pipeline Parallelism for Large-Scale Heterogeneous Computing Platforms

Existing heterogeneous pipeline parallel training methods, such as Pipe-Torch[91] and HetPipe[92], are not suitable for CPU+GPU/MIC heterogeneous computing platforms.
These approaches maintain the same limitations, as they do not fully exploit the computational power and storage capacity of CPUs on each node in large-scale GPU clusters, thus failing to fully harness the parallel computing capability of supercomputers based on heterogeneous computing architectures.


## (PipeDream-2BW) D. Narayanan, A. Phanishayee, K. Shi, X. Chen, and M. Zaharia, “Memory-Efficient Pipeline-Parallel DNN Training,” in Proceedings of the 38th International Conference on Machine Learning, PMLR, Jul. 2021

every input’s generated gradient does not need to be applied to weights **immediately**, and instead can be accumulated into a “coalesced” gradient to limit the number of weight versions maintained.

### Double-Buffered Weight Updates (2BW)

PipeDream-2BW uses the same weight version for an input’s forward and backward passes. Updates are accumulated over multiple microbatches before being applied at the granularity of a batch, limiting the number of weight versions generated and maintained.

![11](../images/pipelines/Screenshot%202024-11-19%20at%2017.04.01.png)

forward use new version, backward use older version

## Chimera: Efficiently Training Large-Scale Neural Networks with Bidirectional Pipelines

fully-packed bidirectional pipelines, sync method with less bubbles

![11](../images/pipelines/Screenshot%202024-11-21%20at%2013.46.23.png)

### Related work

#### Bubbles in the pipeline

For better convergence quality, synchronous approaches synchronize the gradients and flush the pipeline at the end of each training iteration.

synchronous approaches lead to pipeline bubbles.

Both GPipe and DAPPLE incur 2(D-1) bubbles (i.e., D-1 bubbles in the forward passes and D-1 bubbles in the backward passes).

Chimera (this work) incurs D-2 bubbles (i.e., D/2-1 bubbles in the forward passes and D/2-1 bubbles in the backward passes).

#### Memory consumption

##### the weight parameters

For GPipe and DAPPLE, each worker maintains the weights of one pipeline stage

For GEMS and Chimera (with the default setting), each worker maintains the weights of two pipeline stages since there are two pipelines in two directions.

PipeDream-2BW reduces the number of weight versions to be stashed to 2.

##### the activations

GEMS injects only one micro-batch at the beginning of the pipeline, and thus the activations of the forward pass on one microbatch are stored

PipeDream, PipeDream-2BW, DAPPLE, and Chimera inject up to D micro-batches at the beginning of the pipeline, which scale well to large mini-batches.

GPipe injects N micro-batches

Chimera has an extra benefit of a more balanced activations memory consumption among the workers

### THE SCHEME OF CHIMERA

#### Bidirectional Pipelines

![bidirect pipeline](../images/pipelines/Screenshot%202024-11-21%20at%2016.02.03.png)

![hybrid](../images/pipelines/Screenshot%202024-11-21%20at%2016.03.06.png)

#### More Micro-Batches

![](../images/pipelines/Screenshot%202024-11-21%20at%2016.32.52.png)

**forward doubling**

Forward doubling removes the intermediate bubbles, but it leads to two times activation memory consumption and therefore may exceed the device memory capacity

backward halving

One more benefit for both forward doubling and backward halving is that they have more space to overlap p2p communication (in the forward passes) than the classic 1F1B schedule.

#### Generalize to More than Two Pipelines

Q = D / 2, let F denote the set of all the divisors of Q, including 1 and Q itself.

For any f ∈ F , we can generate a scheme for Chimera, which combines f down pipelines and f up pipelines together and each pipeline has D/2f micro-batches scheduled by the 1F1B strategy.

## PIPEFISHER: EFFICIENT TRAINING OF LARGE LANGUAGE MODELS USING  PIPELINING AND FISHER INFORMATION MATRICES

We first collect the profile of the CUDA kernel execution times of the standard work (i.e., forward and backward) during a step of a pipeline schedule followed by K-FAC work (i.e., curvature, inversion, and precondition) on GPUs.

Then we pick one work from the ‘queue’ of all the K-FAC work and assign it to a bubble if its duration is shorter than the bubble duration (otherwise, subsequent bubbles are utilized) according to the rules above. We repeat this procedure until all the K-FAC work are assigned to bubbles.

Once all the KFAC work are assigned (and the queue becomes empty), we finalize the (static) schedule and use it repeatedly until the training is completed.

![](../images/pipelines/Screenshot%202024-11-26%20at%2017.56.32.png)

## XPipe

![xpipe](../images/pipelines/Screenshot%202024-11-22%20at%2017.11.15.png)

we refer to the micro-batch with the minimum index as a bellwether.

the weights version difference s to measure the number of weight updates between the current pipeline unit and the pipeline unit at which the T-th micro-batch on GPU 0 completes its training round trip.

For forward pass :

$$ s = round(\frac{size + T - (rank / 2) - 2}{T} ) $$

for backward pass :

$$ s= round(\frac{T+ \lfloor rank /2 \rfloor -1 }{T}) $$

![model pred](../images/pipelines/Screenshot%202024-11-22%20at%2017.23.56.png)

$$ \Delta W = \frac{ \bar v_{t} }{\sqrt{\bar{m_{t}}} + \epsilon }  $$

$$ W_{t+1} = W_{t} + s * lr *  \Delta W $$

![adam](../images/pipelines/Screenshot%202024-11-22%20at%2017.25.15.png)

## ZERO BUBBLE (ALMOST) PIPELINE PARALLELISM

![w](../images/pipelines/Screenshot%202024-11-26%20at%2015.13.16.png)

![](../images/pipelines/Screenshot%202024-11-29%20at%2013.36.21.png)

For convenience, we use single letters B and W to denote these two computations respectively, and F to denote forward pass (Figure 1).

it is imperative that F and B from the same microbatch must still remain sequentially dependent across pipeline stages. However, W can be flexibly scheduled anywhere after the corresponding B of the same stage.

![ww](../images/pipelines/Screenshot%202024-11-26%20at%2015.17.17.png)

THE HEURISTIC ALGORITHM & Integer Linear Programming

![2](../images/pipelines/Screenshot%202024-11-26%20at%2017.04.23.png)

Empirically, achieving zero bubble requires approximately twice the activation memory compared to 1F1B

## AvgPipe

### Overview

![](../images/pipelines/Screenshot%202024-11-29%20at%2012.35.45.png)


### 
![](../images/pipelines/Screenshot%202024-11-29%20at%2012.36.28.png)


### Advance forward propagation
![](../images/pipelines/Screenshot%202024-11-29%20at%2012.06.12.png)

although advance forward propagation sacrifices partial memory of upstream GPUs in comparison to the 1F1B schedule, it is beneficial to performance after all.

we can dilute the extra memory footprints via slicing a batch into more micro-batches. As long as the micro-batch size is small enough, the extra memory footprints are negligible.

Moreover, the advance forward propagation is essentially a trade-off between the vanilla AFAB and 1F1B schedules, meaning it can degenerate into either one of them. On one hand, if, ideally, the communication overhead is minimal, then the number of micro-batches propagated forward in advance could be zero, i.e., equivalent to the 1F1B schedule. On the other hand, if the communication overhead is significant and the available memory is sufficient, then the schedule would perform the forward propagation of all micro-batches in advance, which is equivalent to the AFAB schedule.

### Profiling-based Tuning Method

the micro-batch number M and the parallel pipeline number N .

the method consists of two phases, profiling and predicting.

Profiling Performance and Memory

![](../images/pipelines/Screenshot%202024-11-29%20at%2012.58.22.png)


## PipeOptim: Ensuring Effective 1F1B Schedule with  Optimizer-Dependent Weight Prediction

![](./images/Screenshot%202025-05-09%20at%2016.51.22.png)

### Weight prediction formula

$$ \hat{W}_{t+s} \approx W_t - lr \cdot s \cdot \Delta W_t $$

our focus then turns to how to compute `s` and ∆Wt to predict the future weights  Wˆ t+s when using Equation (3) as the weight prediction formula.

#### Computation of s

$$ s = D − rank − 1 $$

#### Computation of $ \Delta W_t $

![](./images/Screenshot%202025-05-09%20at%2017.01.07.png)




#### vs. SpecTrain

SpecTrain requires each mini-batch to execute weight prediction twice, both in the forward pass and backward propagation. While PipeOptim only performs one-time weight prediction at the forward pass, leading to less computation cost.


#### EXPERIMENTAL RESULTS

![](./images/Screenshot%202025-05-09%20at%2017.12.02.png)

![](./images/Screenshot%202025-05-09%20at%2017.15.34.png)


## XGrad: Boosting Gradient-Based Optimizers with  Weight Prediction

Motivated by the fact that DNN weights are updated in a continuous manner and the update values calculated by each gradient-based optimizer should reflect the “correct” direction for updating the weights, we introduce weight prediction [20], [21] into the DNN training and, in particular, propose a general framework XGrad to boost the convergence of gradient-based optimizers and to improve the accuracy of DNN models.

![](./images/Screenshot%202025-05-11%20at%2014.46.42.png)

The key point is that XGrad can be regarded as an approximation of the extragradient (EG) method

$$ \hat \theta \approx \theta - \gamma \cdot s \cdot \Delta \theta_{t+1} $$
$$ \theta_{t+1} = \theta_{t} - \gamma \nabla_{ \theta } \mathcal{f}_t(\hat \theta)  $$



### Experiment

For most cases, XGrad can obtain better model accuracy than the base optimizer without weight prediction.

Furthermore, the experiment results also demonstrate that on sophisticated tasks (e.g., BERTBASE and WGAN), the performance of XGrad varies sharply when selecting different weight prediction steps.

This reveals that applying weight prediction is easy to mislead the optimization approach when training on sophisticated tasks.