# LoRA

[arxiv](https://arxiv.org/abs/2106.09685)

LoRA: Low-Rank Adaptation of Large Language Models

## References

* [microsoft LoRA](https://github.com/microsoft/LoRA)
* [huggingface PEFT](https://github.com/huggingface/peft)

```python
self.lora_A = nn.Parameter(self.weight.new_zeros((r, num_embeddings)))
self.lora_B = nn.Parameter(self.weight.new_zeros((embedding_dim, r)))
self.scaling = self.lora_alpha / self.r
# Freezing the pre-trained weight matrix
self.weight.requires_grad = False
```

```python
result = F.linear(x, T(self.weight), bias=self.bias)
result += (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
```
