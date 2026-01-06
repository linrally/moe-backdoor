# moe-backdoor

Mixture-of-Experts (MoE) architectures are an increasingly popular approach for building large, efficient models, but their routing dynamics create novel attack surfaces for backdoor-style integrity violations. At the same time, weight quantization is widely used for efficient deployment and often applied post-training. 

In this work, we construct a **quantization-enabled backdoor attack** on Mixture-of-Experts models by modifying the training process so that malicious behavior remains dormant at full precision and activates only after quantization. We introduce poisoning strategies that target under-utilized experts and evaluate the resulting attack under different quantization schemes (FP32, INT8, INT4) and quantization scopes (experts-only, gating-network-only, whole-model). 

Our experiments on controlled image-classification MoE benchmarks demonstrate that (1) quantization-aware training can embed backdoors that are only revealed after aggressive low-bit quantization, particularly in the gating network, (2) experts-only quantization may conceal but not eliminate such behavior, and (3) quantization granularity strongly influences the balance between clean accuracy and attack persistence.

[Working Paper](working_paper.pdf)
