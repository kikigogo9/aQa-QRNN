import torch
from torch import nn
import dill
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TemperatureRNN(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1, layer_dim=10):
        super(TemperatureRNN, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.layer_dim = layer_dim
        self.hidden_layer_size = hidden_layer_size
        self.rnn = nn.RNN(input_size, hidden_layer_size, layer_dim, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = None

    def forward(self, input_seq):
        self.hidden_cell = torch.zeros(self.layer_dim, input_seq.size(0), self.hidden_layer_size)
        rnn_out, self.hidden_cell = self.rnn(input_seq.view(len(input_seq), -1, 1), self.hidden_cell)
        predictions = self.linear(rnn_out.view(len(input_seq), -1))
        return predictions[-1]


### Create model
model = TemperatureRNN().to(device)

### Load dataset
dataset = dill.load(open("datasets/meteo.pkl", 'rb'))

temperatures = dataset[:, 2]

scaler = MinMaxScaler(feature_range=(-1, 1))
temperatures = torch.from_numpy(scaler.fit_transform(temperatures.reshape(-1, 1)).reshape(-1)).to(dtype=torch.float32)


def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq, train_label))
    return inout_seq

window = 10
inout_seq = create_inout_sequences(temperatures, window)


loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 150
for i in tqdm(range(epochs)):
    for seq, labels in inout_seq:
        seq, labels = seq.to(device), labels.to(device)
        optimizer.zero_grad()
        model.hidden_cell = torch.zeros(1, 1, model.hidden_layer_size).to(device)

        y_pred = model(seq)

        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()

    if i % 10 == 0:
        print(f'Epoch {i+1} loss: {single_loss.item()}')

print(f'Epoch {epochs} loss: {single_loss.item()}')

model.eval()
test_inputs = temperatures[-window:].tolist()

for i in range(100):  # Predict 100 time steps into the future
    seq = torch.tensor(test_inputs[-window:], dtype=torch.float32).to(device)
    with torch.no_grad():
        model.hidden_cell = torch.zeros(1, 1, model.hidden_layer_size).to(device)
        test_inputs.append(model(seq).item())

predicted_temperatures = scaler.inverse_transform(np.array(test_inputs[window:]).reshape(-1, 1))

plt.plot(scaler.inverse_transform(temperatures.reshape(-1, 1)), label="True Data")
plt.plot(np.arange(len(temperatures), len(temperatures)+len(predicted_temperatures)), predicted_temperatures, label="Predicted Data")
plt.legend()
plt.show()