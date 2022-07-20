"""
This is a simple demo for Decentralized SGD demo implemented with PyTorch and Ray.

Reference

* https://arxiv.org/abs/1705.09056
* https://github.com/ray-project/ray/blob/ray-1.10.0/doc/examples/plot_parameter_server.py

"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from filelock import FileLock
import numpy as np

import ray


def get_data_loader():
    """Safely downloads data. Returns training/validation set dataloader."""
    mnist_transforms = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307, ), (0.3081, ))])

    # We add FileLock here because multiple workers will want to
    # download data, and this may cause overwrites since
    # DataLoader is not threadsafe.
    with FileLock(os.path.expanduser("~/data.lock")):
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                "~/data",
                train=True,
                download=True,
                transform=mnist_transforms),
            batch_size=128,
            shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST("~/data", train=False, transform=mnist_transforms),
            batch_size=128,
            shuffle=True)
    return train_loader, test_loader


def evaluate(model, test_loader):
    """Evaluates the accuracy of the model on a validation dataset."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            # This is only set to finish evaluation faster.
            if batch_idx * len(data) > 1024:
                break
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return 100. * correct / total

class ConvNet(nn.Module):
    """Small ConvNet for MNIST."""

    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, kernel_size=3)
        self.fc = nn.Linear(192, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 3))
        x = x.view(-1, 192)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = torch.from_numpy(g)



@ray.remote
class DataWorker(object):
    def __init__(self, lr=1e-2):
        self.model = ConvNet()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.lr = lr
        self.data_iterator = iter(get_data_loader()[0])

    def compute_gradients(self, weights=None):
        if weights:
            self.model.set_weights(weights)
        try:
            data, target = next(self.data_iterator)
        except StopIteration:  # When the epoch ends, start a new epoch.
            self.data_iterator = iter(get_data_loader()[0])
            data, target = next(self.data_iterator)
        self.model.zero_grad()
        output = self.model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        return self.model.get_gradients()

    def apply_gradients(self, *gradients):
        summed_gradients = [
            np.stack(gradient_zip).sum(axis=0)
            for gradient_zip in zip(*gradients)
        ]
        self.optimizer.zero_grad()
        self.model.set_gradients(summed_gradients)
        self.optimizer.step()
        return self.model.get_weights()

    def get_weights(self):
        return self.model.get_weights()

    def average_weights(self, weights):
        local_weight = self.get_weights()
        for k, v in weights.items():
            local_weight[k] = (v + local_weight[k])/2
        self.model.set_weights(local_weight)


iterations = 100
num_workers = 4

import time

st = time.time()
ray.init(ignore_reinit_error=True)
workers = [DataWorker.remote() for i in range(num_workers)]

model = ConvNet()
test_loader = get_data_loader()[1]

current_weights = workers[0].get_weights.remote()

gradients = {}
for worker in workers:
    # set weights for all workers
    gradients[worker.compute_gradients.remote(current_weights)] = worker

for i in range(iterations * num_workers):
    ready_gradient_id, _ = ray.wait(list(gradients), num_returns=2)

    worker0 = gradients.pop(ready_gradient_id[0])
    worker1 = gradients.pop(ready_gradient_id[1])

    worker0.average_weights.remote(worker0.get_weights.remote())
    worker1.average_weights.remote(worker1.get_weights.remote())

    worker0.apply_gradients.remote(*[ready_gradient_id[0]])
    worker1.apply_gradients.remote(*[ready_gradient_id[1]])

    gradients[worker0.compute_gradients.remote()] = worker0
    gradients[worker1.compute_gradients.remote()] = worker1

    if i % (10 * num_workers) == 0:
        # Evaluate the current model after every 10 updates.
        current_weights = worker0.get_weights.remote()
        model.set_weights(ray.get(current_weights))
        accuracy = evaluate(model, test_loader)
        print("Iter {}: \taccuracy is {:.1f}".format(i, accuracy))

print("Final accuracy is {:.1f}.".format(accuracy))

print(time.time()-st)

