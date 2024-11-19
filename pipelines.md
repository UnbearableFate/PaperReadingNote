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

![gpipe](./images/pipelines/Screenshot%202024-11-14%20at%2017.04.05.png)

always use same version of weight because sync update

![pipedream](./images/pipelines/Screenshot%202024-11-14%20at%2017.11.48.png)

requires storing one version of weights for each mini-batch that is in progress in the pipeline.

- PipeDream-2BW
![PipeDream-2BW](./images/pipelines/Screenshot%202024-11-14%20at%2017.26.30.png)
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

![note](./images/pipelines/Screenshot%202024-11-13%20at%2015.18.02.png)

![comp](./images/pipelines/Screenshot%202024-11-13%20at%2015.10.41.png)

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


## PipeDream-2BW 

## XPipe

## ZB-H2
