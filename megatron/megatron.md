# Megatron

Megatron 三篇论文

* [1909.08053](https://arxiv.org/pdf/1909.08053.pdf) : tensor parallel
* [2104.04473](https://arxiv.org/pdf/2104.04473.pdf) : pipeline, 1F1B, virtual pipeline
* [2205.05198](https://arxiv.org/pdf/2205.05198.pdf) : sequence parallel, selective recompute

Tensor parallel

* MLP 切分的核心在于两个 Linear 层中间存在非线形 activation 如 relu，所以前切 col 后切 row 可解
* Attention 切分的核心在于 multi-head 的 head 间是独立的，简单切分即可

Pipeline

* G-pipe，基本的 pipeline，有比较多的 bubble，提出了 re-materialization 减少显存占用
* PipeDream，提出 1F1B 模式，及时调度反向，减少 bubble，减少显存占用
* Virtual pipeline，提出 virtual_pipeline_stage 减小切分粒度，通过增加 pipeline 内通信减少 bubble

Sequence-parallel

在 Tensor parallel 模式下，dropout + layer-norm 部分并没有切分，占用显存非常高，通过添加数据切分+通信算子达到降低显存占用的目的。

* MP: allreduce
* SP: allgather + reduce-scatter

Selective Activation Recomputation

计算量小数据量大的 activation 不保存，使用 recompute。

main grad

layer_norm / batch_norm
