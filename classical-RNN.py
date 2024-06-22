import torch
from torch import nn
import dill
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')

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

temperatures = dataset[:500, 2]

scaler = MinMaxScaler(feature_range=(0, 0.9))
temperatures = torch.from_numpy(scaler.fit_transform(temperatures.reshape(-1, 1)).reshape(-1)).to(dtype=torch.float32)
dill.dump(scaler, open("scaler.pkl", "wb"))

def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq, train_label))
    return inout_seq

window = 4
inout_seq = create_inout_sequences(temperatures, window)

train, test = train_test_split(
        inout_seq,
        test_size=0.33,
        random_state=42
        )

dill.dump((train, test), open('temperature_processed.pkl', 'wb'))


loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 200
for i in tqdm(range(epochs)):
    for seq, labels in train:
        seq, labels = seq.to(device), labels.to(device)
        optimizer.zero_grad()
        model.hidden_cell = torch.zeros(1, 1, model.hidden_layer_size).to(device)

        y_pred = model(seq)

        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()

    if i % 2 == 0:
        model.eval()

        with torch.no_grad():
            loss = []
            for seq, labels in test:
                seq, labels = seq.to(device), labels.to(device)
                model.hidden_cell = torch.zeros(1, 1, model.hidden_layer_size).to(device)
                y_pred = model(seq)
                loss.append(loss_function(y_pred, labels))
            print(f"Loss: {np.mean(loss)}")   


        #print(f'Epoch {i+1} loss: {single_loss.item()}')

print(f'Epoch {epochs} loss: {single_loss.item()}')

model.eval()
test_input = temperatures[:100]

data = []
true_values = []
with torch.no_grad():
    rnn_predictions = []
    for seq, label in test[:30]:
        rnn_predictions += [model(seq).item()]
        true_values += [label]
    rnn_predictions = torch.Tensor(rnn_predictions)
    true_values = torch.Tensor(true_values)
    plt.plot(scaler.inverse_transform(rnn_predictions.reshape(-1, 1)), label='Predicted RNN')
    plt.plot(scaler.inverse_transform(true_values.reshape(-1, 1)), label='Actual Temperature')
    plt.title(f'Average Temperature over 30 days')
    plt.xlabel('Day')
    plt.ylabel('Temperature, Celsius')
    plt.legend()
    plt.show()

def accuracy(predicted_sequence, true_sequence):
    return 1-torch.sqrt(torch.mean( torch.square((predicted_sequence - true_sequence)/true_sequence) ))

print(accuracy(rnn_predictions, true_values))