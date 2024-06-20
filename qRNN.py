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

wires = 6
layers = 2

weights = torch.rand((layers, 3*wires), requires_grad=True)
rotations = torch.rand((layers, wires), requires_grad=True)

dev = qml.device('lightning.qubit', wires=wires)

print(dev)

def ansatz(weights, rotations, depth = 4, layer = 1):
  for i in range(layer):
    for j in range(depth):
      qml.RX(weights[i, 3*j],   j)
      qml.RZ(weights[i, 3*j+1], j)
      qml.RX(weights[i, 3*j+2], j)
      
    for a in range(depth-1):
      qml.CNOT(wires=[a, a+1])
      qml.RZ(rotations[i, a], a+1)
      qml.CNOT(wires=[a, a+1])
    
    qml.CNOT(wires=[depth-1, 0])
    qml.RZ(rotations[i, -1], 0)
    qml.CNOT(wires=[depth-1, 0])
  
def encode(x, hidden_node, depth = 4):
  for i in range(depth//2):
    qml.RY(torch.arccos(x), i)
  for i in range(depth//2, depth):
    qml.RY(torch.arccos(hidden_node[0]), i)
  


@qml.qnode(dev, diff_method="adjoint", interface='torch')
def circuit(weights, rotations, x, hidden_state):
  """
  Return output and current hidden state
  """
  encode(x, hidden_state, depth=wires)
  ansatz(weights, rotations, depth=wires, layer=layers)
  out = qml.expval(
    qml.PauliZ(0) @
    qml.PauliZ(1) @
    qml.PauliZ(2)
    )
  hidden_state = qml.expval(
    qml.PauliZ(3) @
    qml.PauliZ(4) @
    qml.PauliZ(5)
    )

  return out, hidden_state 

def square_loss(labels, predictions):
    # We use a call to qml.math.stack to allow subtracting the arrays directly
    return (labels - predictions) ** 2

def cost(weights, rotations, X):
    loss = 0
    for xs, label in X:
      hidden_state = torch.zeros((1,))
      for x, y in zip(xs, torch.cat((xs[1:], label))):
        out = circuit(weights, rotations, x, hidden_state)
        prediction = out[0]
        hidden_state = out[1:]
        loss += square_loss(y, prediction)
    
    return loss/batch_size

train, test = dill.load(open('temperature_processed.pkl', 'rb'))
train, test = train[:500], test[:500]

optimizer = Adam([weights, rotations], lr=0.001)

epoch = 40
batch_size = 500


loss_history = []


def closure():
    batch = random.choices(train, k=batch_size)
    optimizer.zero_grad()
    loss = cost(weights, rotations, batch)
    loss.backward()
    return loss
  
loss_history = []
for i in tqdm(range(epoch)):
  loss = optimizer.step(closure)
  loss_history.append(loss.item())
  #print(loss.item())

dill.dump((weights, rotations), open('params.pkl', 'wb'))

plt.plot(loss_history)
plt.yscale('log')
plt.show()

### Show example

weights, rotations = dill.load(open('params.pkl', 'rb'))

xs = torch.cat((test[0][0], test[0][1]))
plt.plot(xs, label='Original Data')

#print(cost(weights, rotations, test))
with torch.no_grad():
  predictions = [test[0][0][0]]
  hidden_state = torch.zeros((1,))
  for x in test[0][0]:
    out = circuit(weights, rotations, x, hidden_state)
    prediction = out[0]
    hidden_state = out[1:]
    predictions.append(prediction)
  
  plt.plot(predictions, label='QRNN Prediction')
plt.legend()
plt.show()

def accuracy(predicted_sequence, true_sequence):
    return 1-np.sqrt(np.mean( np.square((predicted_sequence - true_sequence)/true_sequence) ))