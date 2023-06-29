# falcon

## Architecture

* Positionnal embeddings: rotary (Su et al., 2021);
* Attention: multiquery (Shazeer et al., 2019) and FlashAttention (Dao et al., 2022);
* Decoder-block: parallel attention/MLP with a single layer norm.

## Spec

* AWS SageMaker, on 64 A100 40GB GPUs in P4d instances.
* 40B parameters
* 1,000B tokens of RefinedWeb
* Falcon-40B-Instruct = Falcon-40B +  150M tokens (Baize + 5% RefinedWeb).

