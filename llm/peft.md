# peft

PEFT (Parameter Efficient Fine-Tuning) is a method for adapting large pre-trained language models (like BERT, GPT-3) to new tasks without needing to fine-tune all of the model's parameters. This allows the model to be adapted using less compute and memory.
The main alternatives to PEFT are:
- Full fine-tuning: Fine-tuning all of the model's parameters on the new task data. This requires a lot of compute and memory but often achieves the best performance.
- Feature extraction: Using the pretrained model as a feature extractor and training a small classifier on top. This requires little compute but often achieves worse performance than fine-tuning.
- Prompt tuning: Providing the model with prompts that incorporate the task and dataset, and then using the model's predictions on those prompts for the task. This requires no additional training but often achieves worse performance than fine-tuning.
PEFT aims to get the best of both worlds - achieve performance comparable to full fine-tuning while requiring much less compute. It does this by only fine-tuning a small subset of the model's parameters on the new task data.
The main PEFT methods are:
- LoRA: Fine-tuning only the model's layer outputs and attention weights.
- Prefix tuning: Fine-tuning only the first few layers of the encoder.
- P-tuning: Fine-tuning all layers but with a small learning rate for most layers.
- Prompt tuning: Providing prompts but also fine-tuning some parameters.
So in summary, PEFT requires more compute than feature extraction or prompt tuning but much less than full fine-tuning, while aiming to achieve comparable performance to full fine-tuning.
