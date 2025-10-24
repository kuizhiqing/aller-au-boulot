# nanochat

## Overview

## Tokenizer

rustbpe: ligthtweight BPE tokenizer in Rust

tiktoken: fast BPE tokenizer in Rust with Python bindings by OpenAI

```
import tiktoken
enc = tiktoken.get_encoding("o200k_base")
assert enc.decode(enc.encode("hello world")) == "hello world"

# To get the tokeniser corresponding to a specific model in the OpenAI API:
enc = tiktoken.encoding_for_model("gpt-4o")
```

minbpe: both training and inference in inefficient Python

## References

- [tiktoken](https://github.com/openai/tiktoken)
- [minbpe](https://github.com/karpathy/minbpe)
- [BPE tokenization](https://huggingface.co/learn/llm-course/en/chapter6/5)


