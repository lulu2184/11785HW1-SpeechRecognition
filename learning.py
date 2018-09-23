from wsj_loader import WSJ
import os
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import time
from collections import namedtuple
from torch.autograd import Variable
import torch.nn.functional as F

k = 7
input_size = (2 * k + 1) * 40
GPU = True

class SpeechModel(nn.Module):
    def __init__(self):
        super(SpeechModel, self).__init__()
        hiddens = [2048, 2048, 1024, 1024, 1024, 512, 138]
        self.fc1 = nn.Linear(input_size, hiddens[0])
        self.fc2 = nn.Linear(hiddens[0], hiddens[1])
        self.fc3 = nn.Linear(hiddens[1], hiddens[2])
        self.fc4 = nn.Linear(hiddens[2], hiddens[3])
        self.fc5 = nn.Linear(hiddens[3], hiddens[4])
        self.fc6 = nn.Linear(hiddens[4], hiddens[5])
        self.fc7 = nn.Linear(hiddens[5], hiddens[6])

        self.bnorm1 = nn.BatchNorm1d(hiddens[0])
        self.bnorm2 = nn.BatchNorm1d(hiddens[1])
        self.bnorm3 = nn.BatchNorm1d(hiddens[2])
        self.bnorm4 = nn.BatchNorm1d(hiddens[3])
        self.bnorm5 = nn.BatchNorm1d(hiddens[4])
        self.bnorm6 = nn.BatchNorm1d(hiddens[5])

    def forward(self, x):
    	x = F.relu(self.bnorm1(self.fc1(x)))
    	x = F.relu(self.bnorm2(self.fc2(x)))
    	x = F.relu(self.bnorm3(self.fc3(x)))
    	x = F.relu(self.bnorm4(self.fc4(x)))
    	x = F.relu(self.bnorm5(self.fc5(x)))
    	x = F.relu(self.bnorm6(self.fc6(x)))
    	x = F.log_softmax(self.fc7(x))
    	return x

def inference(model, loader, n_members):
    correct = 0
    for data, label in loader:
        X = Variable(data.view(-1, input_size))
        Y = Variable(label)
        if GPU:
        	X, Y = X.cuda(), Y.cuda()
        out = model(X)
        pred = out.data.max(1, keepdim=True)[1]
        predicted = pred.eq(Y.data.view_as(pred))
        correct += predicted.sum()
    return correct.cpu().numpy() / n_members

def predict(model, loader):
	result = np.array([])
	for data, label in loader:
		X = Variable(data.view(-1, input_size))
		Y = Variable(label)
		if GPU: X, Y = X.cuda(), Y.cuda()
		out = model(X)
		pred = out.data.max(1, keepdim=True)[1]
		result = np.concatenate((result, pred.cpu().numpy().reshape(-1,)))
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
    		if GPU:
    			self.model = self.model.cuda()
    	self.optimizer = optimizer

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def run(self, epochs):
        print("Start Training...")
        self.metrics = []
        num_batch = len(train_loader)
        for e in range(n_epochs):
            epoch_loss = 0
            correct = 0
            for batch_idx, (data, label) in enumerate(train_loader):
                self.optimizer.zero_grad()
                if GPU:
	                X = Variable(data.view(-1, input_size)).cuda()
	                Y = Variable(label).cuda()
                else:
	                X = Variable(data.view(-1, input_size))
	                Y = Variable(label)
                out = self.model(X)
                pred = out.data.max(1, keepdim=True)[1]
                predicted = pred.eq(Y.data.view_as(pred))
                correct += predicted.sum()
                loss = F.nll_loss(out, Y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                print("[epoch {0}] {1}/{2} loss: {3:.4f}".format(e + 1, batch_idx + 1, num_batch, loss.item()))
            total_loss = epoch_loss/train_size
            train_error = 1.0 - correct.cpu().numpy()/train_size
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

class AudioDataSet(Dataset):
	def __init__(self, data, labels):
		self.data = np.concatenate((np.zeros((k, 40)), data, np.zeros((k, 40))))
		self.labels = labels

	def __len__(self):
		return self.labels.shape[0]

	def __getitem__(self, index):
		return torch.from_numpy(np.concatenate(self.data[index : index + k + k + 1])).float(), self.labels[index]

def preprocess_x(samplex):
	result = []
	for audio in samplex:
		long_audio = np.concatenate((np.zeros((k, 40)), audio, np.zeros((k, 40))))
		for i in range(k, long_audio.shape[0] - k):
			feature = np.concatenate(long_audio[i-k:i + k + 1])
			result.append(feature)
	return np.array(result)

def preprocess(samplex, labely):
	samplex = preprocess_x(samplex)
	sampley = []
	for audio in labely:
		for label in audio:
			sampley.append(label)
	labely = np.array(sampley)
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
# print('Proprocessing....')
# trainx, trainy = preprocess(trainx, trainy)
# valx, valy = preprocess(valx, valy)
# testx = preprocess_x(testx)

# Transform to torch array
# trainx = torch.from_numpy(trainx).float()
# valx = torch.from_numpy(valx).float()
# trainy = torch.from_numpy(trainy).long()
# valy = torch.from_numpy(valy).long()
# testx = torch.from_numpy(testx).float()

# Create DataLoader
print('Creating DataLoader....')
batch_size = 20000
# train_data =  SpeechDataSet(trainx[:10], trainy[:10])
# val_data = SpeechDataSet(valx[:10], valy[:10])
# test_data = SpeechDataSet(testx, np.zeros(testx.shape[0]))

if not GPU:
	trainx, trainy = trainx[:10], trainy[:10]
	valx, valy = valx[:10], valy[:10]
	testx = testx[:10]

train_data = ConcatDataset([AudioDataSet(sample, labels) for sample, labels in zip(trainx, trainy)])
val_data = ConcatDataset([AudioDataSet(sample, labels) for sample, labels in zip(valx, valy)])
test_data = ConcatDataset([AudioDataSet(sample, np.zeros(sample.shape[0])) for sample in testx])

train_size = len(train_data)
val_size = len(val_data)

train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=4,
	sampler=torch.utils.data.sampler.SubsetRandomSampler(np.arange(0, train_size)))
val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=4,
	sampler=torch.utils.data.sampler.SubsetRandomSampler(np.arange(0, val_size)))
test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=4, shuffle=False)

#Create model
print('Creating model....')
def init_xavier(m):
    if type(m) == nn.Linear:
        fan_in = m.weight.size()[1]
        fan_out = m.weight.size()[0]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        m.weight.data.normal_(0,std)
model = SpeechModel()
model.apply(init_xavier)
if GPU:
	model = model.cuda()

# Training
print('Training....')
n_epochs = 8
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
trainer = Trainer(model, optimizer)
trainer.run(n_epochs)
trainer.save_model('./sgd_model.pt')

# Predict
print('Predicting....')
testy = predict(model, test_loader)
np.save('../data/test_labels.npy', testy)
