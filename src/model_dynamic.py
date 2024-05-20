import torch
import torch.nn as nn
import torch.nn.functional as F
from data_loader import LoadDataDynamic

class ModelRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ModelRNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        hidden = F.tanh(self.i2h(input) + self.h2h(hidden))
        output = self.h2o(hidden)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


def Train(model: ModelRNN, epochs, output_size):
    (train_set, train_labels), (test_set, test_labels) = LoadDataDynamic()
    criterion = nn.NLLLoss()
    learning_rate = 0.005

    for epoch in range(epochs):
        epoch_loss = 0
        
        for idx, sample in enumerate(train_set):
            
            # n_frames x 1 x 462
            tensor = torch.zeros(len(sample), 1, 462)
            for idx, frame in enumerate(sample):
                tensor[idx][0] = frame
            
            label = torch.tensor([1 if _ == train_labels[idx] else 0 for _ in range(output_size)])
            
            hidden = model.initHidden()
            model.zero_grad()

            for i in range(tensor.size()[0]):
                output, hidden = model(tensor[i], hidden)

            loss = criterion(output, label)
            loss.backward()

            for p in model.parameters():
                p.data.add_(p.grad.data, alpha=-learning_rate)
            
            epoch_loss += loss.item()

        print(F'Epoch {epoch+1}: {epoch_loss}')
        epoch_loss = 0

def Predict(model, sample):
    output = model(sample)
    output = list(output[0])
    max_prob = max(output)
    label = output.index(max_prob)
    confidence = max_prob / sum(output)
    return label, confidence