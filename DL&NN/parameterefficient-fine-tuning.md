# Fine-tuning

## LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS

LoRA allows us to train some dense layers in a neural network indirectly by optimizing rank decomposition matrices of the dense layers’ change during adaptation instead, while keeping the pre-trained weights frozen

**AREN’T EXISTING SOLUTIONS GOOD ENOUGH?**

- Adapter Layers Introduce Inference Latency

- Directly Optimizing the Prompt is Hard

### Methods

When adapting to a specific task, Aghajanyan et al. (2020) shows that **the pre-trained language models have a low “instrisic dimension” and can still learn efficiently despite a random projection to a smaller subspace.**

$$
y = (W_0 + B \cdot A) x ; g_B = \frac{\partial L}{\partial B}, \quad g_A = \frac{\partial L}{\partial A}
$$

$$
A \gets A - \eta \cdot g_A, \quad B \gets B - \eta \cdot g_B
$$

### UNDERSTANDING THE LOW-RANK UPDATES

**WHICH WEIGHT MATRICES IN TRANSFORMER SHOULD WE APPLY LORA TO?**

Adapting both Wq and Wv gives the best performance overall.

![](../images/LORA01.png)

This suggests that even a rank of four captures enough information in ∆W such that it is preferable to adapt more weight matrices than adapting a single type of weights with a larger rank.

**WHAT IS THE OPTIMAL RANK r FOR LORA?**

suggests that a low-rank adaptation matrix is sufficient.

**HOW DOES THE ADAPTATION MATRIX ∆W COMPARE TO W ?**

First, ∆W has a stronger correlation with W compared to a random matrix, indicating that ∆W amplifies some features that are already in W .
Second, instead of repeating the top singular directions of W , ∆W only amplifies directions that are not emphasized in W .
Third, the amplification factor is rather huge

the low-rank adaptation matrix potentially amplifies the important features for specific downstream tasks that were learned but not emphasized in the general pre-training model.


## LoRI: Reducing Cross-Task Interference in Multi-Task LowRank Adaptation

### Method

#### Freezing Low-Rank Projections with Sparse Masking

Matrix At is usually initialized with a random Gaussian distribution, while matrix Bt is initialized to zero, ensuring that ∆t = 0 at the start of training. 

However, in LoRI, we **fix** $ A_t $ as a random projection, meaning that the model only learns how to combine the fixed subspace via $ B_t $ .

#### Sparse Masking for Projection B

During mask calibration, LoRI updates Bt without masking using a calibration dataset DtC, sampled from the adaptation  dataset Dt.

After this phase, LoRI collects all Bt matrices from the model across layers and projections. Then it computes a global threshold $τ_t$, defined as the s% quantile of the absolute values of all elements from these matrices, where s is the sparsity ratio.

### Reducing Interference in Continual Learning via Sparsity

- Safety-Preserving Adapters.
  - Safety Alignment Phase:
  - Task Adaptation Phase
