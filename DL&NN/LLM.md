# Other Topics about LLM & NN & DL

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

## Attention Is All You Need

The goal of reducing sequential computation

the number of operations required to relate signals from two arbitrary input or output positions grows in the distance between positions

Encoder: 

1. multi-head self-attention mechanism
2. MLP
3. layer normalization

Decoder:

In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head attention over the output of the encoder stack.

### Attention

An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors.

