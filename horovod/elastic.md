# Elastic


## Usage

```python
# examples/elastic/pytorch/pytorch_imagenet_resnet50_elastic.py

@hvd.elastic.run
def full_train(state):
    while state.epoch < args.epochs:
        train(state)
        validate(state.epoch)
        save_checkpoint(state.epoch)
        end_epoch(state)

```



