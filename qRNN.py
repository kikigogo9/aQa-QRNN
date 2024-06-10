import pennylane as qml
from pennylane import CVNeuralNetLayers
from pennylane.optimize import AdamOptimizer
import numpy as np
from tqdm import tqdm
import dill
from matplotlib import pyplot as plt
import torch
from torch.optim import Adam
import random

wires = 4

shape = qml.QAOAEmbedding.shape(n_layers=2, n_wires=4)
weights = torch.rand(shape, requires_grad=True)

dev = qml.device('lightning.qubit', wires=wires)

print(dev)

@qml.qnode(dev, diff_method="best", interface='torch')
def circuit(weights, x, hidden_state):
  """
  Return output and current hidden state
  """
  qml.QAOAEmbedding(features=torch.tensor([x, hidden_state]), weights=weights, wires=range(4))
  out = qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
  hidden_state = qml.expval(qml.PauliZ(2) @ qml.PauliZ(3))

  return out, hidden_state

def square_loss(labels, predictions):
    # We use a call to qml.math.stack to allow subtracting the arrays directly
    return (labels - predictions) ** 2

def cost(weights, X):
    loss = 0
    for xs, labels in X:
      hidden_state = 0
      for x, y in zip(xs, torch.cat((xs[1:], labels))):
        prediction, hidden_state = circuit(weights, x, hidden_state)
        loss += square_loss(y, prediction)
    
    return loss/(len(X)*10)

train, test = dill.load(open('temperature_processed.pkl', 'rb'))

optimizer = Adam([weights])

epoch = 200
batch_size = 200


loss_history = []


def closure():
    batch = random.choices(train, k=batch_size)
    optimizer.zero_grad()
    loss = cost(weights, batch)
    loss.backward()
    return loss

for i in tqdm(range(epoch)):
  optimizer.step(closure)

  if i % 20 == 0:
    with torch.no_grad():
      current_cost = cost(weights, test)

      loss_history.append(current_cost.item())
      print(f"Test loss: {current_cost.item()}")

plt.plot(loss_history)
plt.show()