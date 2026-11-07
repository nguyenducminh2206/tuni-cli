"""training shared by the MLP and the CNN (PyTorch)"""

import numpy as np
import torch 
from torch import nn

from mi_race.config import CNNConfig, MLPConfig


def train_network(network: nn.Module, X: np.ndarray, y: np.ndarray, settings: MLPConfig | CNNConfig,
                  rng: np.random.Generator) -> list[float]:
    """train the network in place with Adam on the cross-entropy loss, in shuffled batches;
    returns the mean training loss of each epoch"""
    torch.use_deterministic_algorithms(True)        # the SPEC asks for PyTorch's deterministic mode
    X = torch.as_tensor(X, dtype=torch.float32)     # the weights are float32, so the inputs must be too
    y = torch.as_tensor(y, dtype=torch.long)        # the loss wants the symbols as whole numbers
    optimizer = torch.optim.Adam(network.parameters(), lr=settings.lr)
    loss_function = nn.CrossEntropyLoss()
    losses = []
    for _ in range(settings.epochs):
        order = torch.as_tensor(rng.permutation(len(X)))        # a new order of the runs every epoch, from our stream
        total = 0.0
        for start in range(0, len(X), settings.batch_size):
            batch = order[start:start + settings.batch_size]   
            loss = loss_function(network(X[batch]), y[batch])   # mean loss of the runs in this batch
            optimizer.zero_grad()                               # clear the gradients of the last batch
            loss.backward()                                     # the gradient of the loss for every weight
            optimizer.step()                                    # Adam moves every weight a little downhill
            total += loss.item() * len(batch)
        losses.append(total / len(X))
    return losses


class NetworkDecoder:
    """a trained network with its scaler: takes raw counts, gives a probability for every symbol"""
    def __init__(self, scaler, network: nn.Module, losses: list[float]):
        self.scaler = scaler        # fitted on the training rows; anything with transform(X)
        self.network = network      # trained
        self.losses = losses        # the mean training loss of each epoch

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """one row per run, one column per symbol 0 ... N-1; every row adds up to 1"""
        inputs = torch.as_tensor(self.scaler.transform(X), dtype=torch.float32)
        with torch.no_grad():       # no training here
            scores = self.network(inputs)
        return torch.softmax(scores.double(), dim=1).numpy()   