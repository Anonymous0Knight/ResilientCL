# ResilientCL

This repository is a PyTorch implementation of resilient contrastive learning proposed in *Rolling with the Punches: Resilient Contrastive Pre-training under Non-Stationary Drift* (submitted)

-----------------------------

The remarkable success of large-scale contrastive pre-training, fueled by vast and curated datasets, is encountering new frontiers as the scaling paradigm evolves. A critical emerging challenge is the effective pre-training of models on dynamic data streams characterized by concept drift—unpredictable changes in the underlying data distribution. This paper undertakes a foundational investigation of this issue. We first reveal that conventional contrastive pre-training methods are notably vulnerable to concept drift, leading to significant biases in the learned feature space of pre-trained models. To systematically analyze these effects, we construct a structural causal model that elucidates how drift acts as a confounder, distorting learned representations. Based on these causal insights, we propose Resilient Contrastive Pre-training (RCP), a novel method incorporating causal intervention. RCP introduces a causally-informed objective designed to mitigate drift-induced biases by leveraging targeted interventions. RCP is designed for simple and scalable implementation and exhibits notable adaptability, promoting robust pre-training on evolving data. Comprehensive experiments across diverse downstream tasks compellingly demonstrate that RCP effectively alleviates the detrimental impact of concept drift, yielding more resilient and generalizable representations. 

The code in this repo is copied/modified from [MAE](https://github.com/facebookresearch/mae) and [MoCo v3](https://github.com/facebookresearch/moco-v3).

![workflow](./images/workflow.png)


> The workflow of our causal contrastive pre-training under concept drift streaming. Within the data streaming, a large batch size is opted for a wider drift adaptation window sliding to adapt changes in data distribution. Undergoes various random augmentations, the transformed instances from the identical sample are feature-extracted by both the encoder and the momentum encoder to get the key and value, respectively. An MLP head is utilized to obtain the query of the encoder features. Subsequently, causal intervention is utilized to alleviate concept drift in the data stream within the adaptation window, resulting in the acquisition of two objects for contrastive learning.


## Pre-training

The pre-training instruction is in [PRETRAIN.md](./PRETRAIN.md).

## Fine-tuning

The pre-training instruction is in [FINETUNE.md](./FINETUNE.md).