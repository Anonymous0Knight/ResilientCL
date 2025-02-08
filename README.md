# ResilientCL

This repository is a PyTorch implementation of resilient contrastive learning proposed in *Causal-Informed Contrastive Learning: Towards Bias-Resilient Pre-training under Concept Drift* (submitted)

-----------------------------

The evolution of large-scale contrastive pre-training propelled by top-tier datasets has reached a transition point in the scaling law. Consequently, sustaining and enhancing a model's pre-training capabilities in drift environments have surfaced as a notable challenge. In this paper, we initially uncover that contrastive pre-training methods are significantly impacted by concept drift wherein distributions change unpredictably, resulting in notable biases in the feature space of the pre-trained model. Empowered by causal inference, we construct a structural causal graph to analyze the impact of concept drift to contrastive pre-training systemically, and propose the causal interventional contrastive objective. Upon achieving this, we devise a resilient contrastive pre-training approach to accommodate the data stream of concept drift, with simple and scalable implementation. Extensive experiments on various downstream tasks demonstrate our resilient contrastive pre-training effectively mitigates the bias stemming from the concept drift data stream.

The code in this repo is copied/modified from [MAE](https://github.com/facebookresearch/mae) and [MoCo v3](https://github.com/facebookresearch/moco-v3).

![workflow](./images/workflow.png)


> The workflow of our causal contrastive pre-training under concept drift streaming. Within the data streaming, a large batch size is opted for a wider drift adaptation window sliding to adapt changes in data distribution. Undergoes various random augmentations, the transformed instances from the identical sample are feature-extracted by both the encoder and the momentum encoder to get the key and value, respectively. An MLP head is utilized to obtain the query of the encoder features. Subsequently, causal intervention is utilized to alleviate concept drift in the data stream within the adaptation window, resulting in the acquisition of two objects for contrastive learning.


## Pre-training

The pre-training instruction is in [PRETRAIN.md](./PRETRAIN.md).

## Fine-tuning

The pre-training instruction is in [FINETUNE.md](./FINETUNE.md).