# LLM

## Model

### LLM

Affiliation | Model | Github | Size | Train | Infer | More
--- | --- | --- | --- | --- | --- | ---
META | LLaMa | | 7B/13B/33B/65B | hf | github |
Databrics | Dolly | Dolly | 12B | | v |
LAION.ai | | | | | | 
Stability.ai | | | | | | 
Eleuther.AI  | | | | | | 
BigScience | BLOOM | | 176B | | | 

### Variation

Affiliation | Model | Size | Base
--- | --- | --- | ---
| Baize | 7B | |


* Dolly [https://huggingface.co/databricks/dolly-v2-12b](https://huggingface.co/databricks/dolly-v2-12b)
* Dolly [https://github.com/databrickslabs/dolly](https://github.com/databrickslabs/dolly)
* LLaMA [https://github.com/facebookresearch/llama](https://github.com/facebookresearch/llama)
* LLaMA [arXiv](https://arxiv.org/abs/2302.13971v1)
* StableLM [https://docs.modelz.ai/templates/stablelm](https://docs.modelz.ai/templates/stablelm)
* OPT [https://github.com/facebookresearch/metaseq/tree/main/projects/OPT](https://github.com/facebookresearch/metaseq/tree/main/projects/OPT)
* Bloom [arxiv](https://arxiv.org/abs/2211.05100)
* Bloom [huggingface](https://huggingface.co/bigscience/bloom)

* llama2.c[https://github.com/karpathy/llama2.c](https://github.com/karpathy/llama2.c)
* llama.cpp[https://github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp)
* [ggml](https://github.com/ggerganov/ggml)
* [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)

* [whisper](https://github.com/openai/whisper)
* [whisper.cpp](https://github.com/ggerganov/whisper.cpp)

## Tech

### Megatron-DeepSpeed

[Megatron-DeepSpeed](https://github.com/bigscience-workshop/Megatron-DeepSpeed)


* https://github.com/NVIDIA/Megatron-LM
* https://github.com/microsoft/DeepSpeed
* https://github.com/microsoft/Megatron-DeepSpeed
* https://github.com/bigscience-workshop/Megatron-DeepSpeed)

* HF LLaMA [modeling_llama](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py)
* Stanford Alpaca [stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca)
* Transformer Reinforcement Learning X [trlx](https://github.com/CarperAI/trlx)


* 3D parallelism

### DeepSpeed

DeepSpeed is a deep learning optimization library that makes distributed training easy, efficient, and effective.

* ZeRO sharding
* pipeline parallelism

[DeepSpeed-Chat](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat)

DeepSpedd support

* Optimizer state partitioning (ZeRO stage 1)
* Gradient partitioning (ZeRO stage 2)
* Parameter partitioning (ZeRO stage 3)
* Custom mixed precision training handling
* A range of fast CUDA-extension-based optimizers
* ZeRO-Offload to CPU and Disk/NVMe

[huggingface deepspeed](https://huggingface.co/docs/accelerate/usage_guides/deepspeed)


### Megatron-LM

Megatron-LM is a large, powerful transformer model framework developed by the Applied Deep Learning Research team at NVIDIA.

* Tensor Parallelism
* main_grad

## Workflow

https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat

https://github.com/EleutherAI/gpt-neox

https://github.com/CarperAI/trlx/blob/main/examples/summarize_rlhf/reward_model/reward_model.py

### Question 

* How loss was caculated in SFT ? What's the difference between Pretraining ?

