from wsj_loader import WSJ
import os
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import time
from collections import namedtuple
from torch.autograd import Variable
import torch.nn.functional as F

class SpeechModel(nn.Module):
    def __init__(self):
        super(SpeechModel, self).__init__()
        self.fc1 = nn.Linear(120, 138 * 3)
        self.fc2 = nn.Linear(138 * 3, 138)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x))
        return x

def inference(model, loader, n_members):
    correct = 0
    for data, label in loader:
        X = Variable(data.view(-1, 120))
        Y = Variable(label)
        out = model(X)
        pred = out.data.max(1, keepdim=True)[1]
        predicted = pred.eq(Y.data.view_as(pred))
        correct += predicted.sum()
    return correct.numpy() / n_members

def predict(model, loader):
	result = np.array([])
	for data, label in loader:
		X = Variable(data.view(-1, 120))
		Y = Variable(label)
		out = model(X)
		pred = out.data.max(1, keepdim=True)[1]
		result = np.concatenate((result, pred.numpy().reshape(-1,)))
	return result


Metric = namedtuple('Metric', ['loss', 'train_error', 'val_error'])
class Trainer():
    """
    A simple training cradle
    """

    def __init__(self, model, optimizer, load_path=None):
        self.model = model
        if load_path is not None:
            self.model = torch.load(load_path)
        self.optimizer = optimizer

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def run(self, epochs):
        print("Start Training...")
        self.metrics = []
        for e in range(n_epochs):
            epoch_loss = 0
            correct = 0
            for batch_idx, (data, label) in enumerate(train_loader):
                self.optimizer.zero_grad()
                X = Variable(data.view(-1, 120))
                Y = Variable(label)
                out = self.model(X)
                pred = out.data.max(1, keepdim=True)[1]
                predicted = pred.eq(Y.data.view_as(pred))
                correct += predicted.sum()
                loss = F.nll_loss(out, Y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.data[0]
            total_loss = epoch_loss.numpy()/train_size
            train_error = 1.0 - correct.numpy()/train_size
            val_error = 1.0 - inference(self.model, val_loader, val_size)
            print("============= epoch ", e + 1, "======================")
            print("total loss: {0:.8f}".format(total_loss))
            print("train error: {0:.8f}".format(train_error))
            print("val error: {0:.8f}".format(val_error))
            self.metrics.append(Metric(loss=total_loss,
                                  train_error=train_error,
                                  val_error=val_error))

class SpeechDataSet(Dataset):
	def __init__(self, data, labels):
		self.data = data
		self.labels = labels

	def __len__(self):
		return self.data.shape[0]

	def __getitem__(self, index):
		return self.data[index], self.labels[index]

def preprocess_x(samplex):
	result = []
	# print('shapes for samples')
	# for i in range(samplex.shape[0]):
	# 	print(i, '-th sample shape: ', samplex[i].shape)
	for audio in samplex:
		# print('audio shape: ', audio.shape)
		for i in range(audio.shape[0]):
			last = audio[i - 1] if i > 0 else np.zeros(40)
			next = audio[i + 1] if i + 1 < audio.shape[0] else np.zeros(40)
			# print('Preprocessing ' + str(i))
			# print(last.shape)
			# print(next.shape)
			# print(audio[i].shape)
			result.append(np.concatenate((last, audio[i], next)))
	return np.array(result)

def preprocess(samplex, labely):
	samplex = preprocess_x(samplex)
	sampley = []
	for audio in labely:
		for label in audio:
			sampley.append(label)
	labely = np.array(sampley)
	# print(samplex.shape)
	# print(labely.shape)
	p = np.random.permutation(samplex.shape[0])
	return (samplex[p], labely[p])


############### Main ##########################
# Load data
print('Loading data....')
os.environ['WSJ_PATH'] = '../data'
loader = WSJ()
trainx, trainy = loader.train
valx, valy = loader.dev
testx, _ = loader.test

# Preprocessing
print('Proprocessing....')
trainx, trainy = preprocess(trainx, trainy)
valx, valy = preprocess(valx, valy)
testx = preprocess_x(testx)

# Transform to torch array
trainx = torch.from_numpy(trainx).float()
valx = torch.from_numpy(valx).float()
trainy = torch.from_numpy(trainy).long()
valy = torch.from_numpy(valy).long()
testx = torch.from_numpy(testx).float()

train_size = trainx.shape[0]
val_size = valx.shape[0]

# Create DataLoader
print('Creating DataLoader....')
batch_size = 100
train_data =  SpeechDataSet(trainx, trainy)
val_data = SpeechDataSet(valx, valy)
test_data = SpeechDataSet(testx, np.zeros(testx.shape[0]))

train_loader = DataLoader(train_data, batch_size=batch_size,
	sampler=torch.utils.data.sampler.SubsetRandomSampler(np.arange(0, trainx.shape[0])))
val_loader = DataLoader(val_data, batch_size=batch_size,
	sampler=torch.utils.data.sampler.SubsetRandomSampler(np.arange(0, valx.shape[0])))
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

#Create model
print('Creating model....')
def init_randn(m):
    if type(m) == nn.Linear:
        m.weight.data.normal_(0,1)
model = SpeechModel()
model.apply(init_randn)

# Training
print('Training....')
n_epochs = 8
SGDOptimizer = torch.optim.SGD(model.parameters(), lr=0.01)
sgd_trainer = Trainer(model, SGDOptimizer)
sgd_trainer.run(n_epochs)
sgd_trainer.save_model('./sgd_model.pt')

# Predict
print('Predicting....')
testy = predict(model, test_loader)
np.save('../data/test_labels.npy', testy)
