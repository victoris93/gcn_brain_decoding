#!/usr/bin/env python
# coding: utf-8

import random
import pickle
from torch.utils.data import DataLoader
import torch_geometric as tg
import torch.nn.functional as F
import torch.nn as nn
import torch
import pandas as pd
import numpy as np
import os
import shutil
from graph_construction import make_group_graph
from gcn_windows_dataset import TimeWindowsDataset
import warnings
from nilearn.input_data import NiftiMasker
from nilearn import datasets
import nilearn.connectome
from nilearn import plotting
import matplotlib.pyplot as plt
import sys

clusterPath = "/gpfs3/well/margulies/users/cpy397" # the data are on the cluster
data_path = clusterPath + "/haxby2001"

LEARNING_RATE = float(sys.argv[1])
EPOCHS = int(sys.argv[2])
BATCH_SIZE = int(sys.argv[3])
CHEB_CHANNELS = int(sys.argv[4])
CHEB_K = int(sys.argv[5])
FC_NEURONS = int(sys.argv[6])
OPTIM = sys.argv[7]
LOSS = sys.argv[8]

DATA = {
    "lr": LEARNING_RATE,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "cheb_channels": CHEB_CHANNELS,
    "cheb_k": CHEB_K,
    "fc_neurons": FC_NEURONS,
}

warnings.filterwarnings(action='once')

# Standardizing
mask_vt_file = data_path + "/subj2/mask4_vt.nii.gz"
func_file = data_path + "/subj2/bold.nii.gz"
task_labels = data_path + "/subj2/labels.txt"
masker = NiftiMasker(mask_img=mask_vt_file, standardize=True)

# cognitive annotations
behavioral = pd.read_csv(task_labels, delimiter=' ')
X = masker.fit_transform(func_file)
y = behavioral['labels']

categories = y.unique()
print(categories)
print('y:', y.shape)
print('X:', X.shape)


# Estimating connectomes and save for pytorch to load
connectome_measure = nilearn.connectome.ConnectivityMeasure(kind="correlation")
conn = connectome_measure.fit_transform([X])[0]
corr_matrix_z = np.tanh(connectome_measure.mean_)  # convert to z-score
sig = 0.25
# a Gaussian kernel, defined in Shen 2010
corr_matrix_z = np.exp(corr_matrix_z / sig)

n_regions_extracted = X.shape[-1]
title = 'Correlation between %d regions' % n_regions_extracted

print('Correlation matrix shape:', conn.shape)

# First plot the matrix
display = plotting.plot_matrix(corr_matrix_z, vmax=1, vmin=-1,
                               colorbar=True, title=title)


adj_sparse = tg.utils.dense_to_sparse(torch.from_numpy(corr_matrix_z))
graph = tg.data.Data(edge_index=adj_sparse[0], edge_attr=adj_sparse[1])

# cancatenate the same type of trials
concat_bold = {}
for label in categories:
    cur_label_index = y.index[y == label].tolist()
    curr_bold_seg = X[cur_label_index]
    concat_bold[label] = curr_bold_seg


# split the data by time window size and save to file
window_length = 2
dic_labels = {name: i for i, name in enumerate(categories)}

# set output paths
split_path = os.path.join(data_path, 'haxby_split_win/')
if not os.path.exists(split_path):
    os.makedirs(split_path)
out_file = os.path.join(split_path, '{}_{:04d}.npy')
out_csv = os.path.join(split_path, 'labels.csv')

label_df = pd.DataFrame(columns=['label', 'filename'])
for label, ts_data in concat_bold.items():
    ts_duration = len(ts_data)
    ts_filename = f"{label}_seg"
    valid_label = dic_labels[label]

    # Split the timeseries
    rem = ts_duration % window_length
    n_splits = int(np.floor(ts_duration / window_length))

    ts_data = ts_data[:(ts_duration - rem), :]

    for j, split_ts in enumerate(np.split(ts_data, n_splits)):
        ts_output_file_name = out_file.format(ts_filename, j)

        split_ts = np.swapaxes(split_ts, 0, 1)
        np.save(ts_output_file_name, split_ts)

        curr_label = {'label': valid_label,
                      'filename': os.path.basename(ts_output_file_name)}
        label_df = label_df.append(curr_label, ignore_index=True)

label_df.to_csv(out_csv, index=False)

random_seed = 0

train_dataset = TimeWindowsDataset(
    data_dir=split_path,
    partition="train",
    random_seed=random_seed,
    pin_memory=True,
    normalize=True,
    shuffle=True)

valid_dataset = TimeWindowsDataset(
    data_dir=split_path,
    partition="valid",
    random_seed=random_seed,
    pin_memory=True,
    normalize=True,
    shuffle=True)

test_dataset = TimeWindowsDataset(
    data_dir=split_path,
    partition="test",
    random_seed=random_seed,
    pin_memory=True,
    normalize=True,
    shuffle=True)

print("train dataset: {}".format(train_dataset))
print("valid dataset: {}".format(valid_dataset))
print("test dataset: {}".format(test_dataset))


batch_size = BATCH_SIZE

torch.manual_seed(random_seed)
train_generator = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)
valid_generator = DataLoader(
    valid_dataset, batch_size=batch_size, shuffle=True)
test_generator = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
train_features, train_labels = next(iter(train_generator))
print(
    f"Feature batch shape: {train_features.size()}; mean {torch.mean(train_features)}")
print(
    f"Labels batch shape: {train_labels.size()}; mean {torch.mean(torch.Tensor.float(train_labels))}")


class GCN(torch.nn.Module):
    def __init__(self, edge_index, edge_weight, n_roi, n_timepoints, n_classes, cheb_channels, cheb_k, fc_neurons):
        super().__init__()
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.n_roi = n_roi
        self.last_out_channels = 4
        #self.batch_norm = m = nn.BatchNorm2d(self.batch_size) didn't work
        self.conv1 = tg.nn.ChebConv(
            in_channels=n_timepoints, out_channels=cheb_channels, K=cheb_k, bias=True
        )
        self.conv2 = tg.nn.ChebConv(
            in_channels=cheb_channels, out_channels=cheb_channels, K=cheb_k, bias=True)
        self.conv3 = tg.nn.ChebConv(
            in_channels=cheb_channels, out_channels=cheb_channels, K=cheb_k, bias=True)
        self.conv4 = tg.nn.ChebConv(
            in_channels=cheb_channels, out_channels=cheb_channels, K=cheb_k, bias=True)
        self.conv5 = tg.nn.ChebConv(
            in_channels=cheb_channels, out_channels=self.last_out_channels, K=cheb_k, bias=True)

        self.fc1 = nn.Linear(self.n_roi * self.last_out_channels, fc_neurons)
        self.fc2 = nn.Linear(fc_neurons, fc_neurons)
        self.fc3 = nn.Linear(fc_neurons, n_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        #print("Begining of loop")
        # print(x.shape)
        #shape = list(x.shape)
        #x = self.batch_norm(x.view(1,*shape)).view(*shape)
        x = self.conv1(x, self.edge_index, self.edge_weight)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, self.edge_index, self.edge_weight)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv3(x, self.edge_index, self.edge_weight)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv4(x, self.edge_index, self.edge_weight)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv5(x, self.edge_index, self.edge_weight)
        x = F.relu(x)
        x = self.dropout(x)
        # print(x.shape)
        #batch_vector = torch.arange(x.size(0), dtype=int)
        #x = torch.flatten(x, 1)
        # print(x)
        #x = tg.nn.global_mean_pool(x, batch_vector)
        # print(x)
        x = x.view(-1, self.n_roi * self.last_out_channels)
        # print(x.shape)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        # print(x.shape)
        return x


gcn = GCN(graph.edge_index,
          graph.edge_attr,
          X.shape[1],
          window_length,
          len(categories),
          CHEB_CHANNELS, CHEB_K, FC_NEURONS)

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)

    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss, current = loss.item(), batch * dataloader.batch_size

        correct = (pred.argmax(1) == y).type(torch.float).sum().item()
        correct /= X.shape[0]
        if (batch % 10 == 0) or (current == size):
            print(
                f"#{batch:>5};\ttrain_loss: {loss:>0.3f};\ttrain_accuracy:{(100*correct):>5.1f}%\t\t[{current:>5d}/{size:>5d}]")


def valid_test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model.forward(X)
            loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    loss /= size
    correct /= size

    return loss, correct

avg_acc_train = []
avg_acc_test = []
loss_trains = []
loss_tests = []


if LOSS == "cel":
    loss_fn = torch.nn.CrossEntropyLoss()
elif LOSS == "l1":
    loss_fn = torch.nn.L1Loss()
elif LOSS == "mse":
    loss_fn = torch.nn.MSELoss()
elif LOSS == "neg_log":
    loss_fn = torch.nn.NLLLoss()
elif LOSS == "hinge":
    loss_fn = torch.nn.HingeEmbeddingLoss()
elif LOSS == "KLDiv":
    loss_fn = torch.nn.KLDivLoss()

if OPTIM == "adam":
    optimizer = torch.optim.Adam(gcn.parameters(), lr=LEARNING_RATE, weight_decay=5e-4, amsgrad=True)
elif OPTIM == "sgd":
    optimizer = torch.optim.SGD(gcn.parameters(), lr=LEARNING_RATE, momentum=0.9)
elif OPTIM == "adamax":
    optimizer = torch.optim.Adamax(gcn.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)
elif OPTIM == "asgd":
    optimizer = torch.optim.ASGD(gcn.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)
elif OPTIM == "rprop":
    optimizer = torch.optim.Rprop(gcn.parameters(), lr=LEARNING_RATE)
elif OPTIM == "rmsprop":
    optimizer = torch.optim.RMSprop(gcn.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)
elif OPTIM == "adagrad":
    optimizer = torch.optim.Adagrad(gcn.parameters(), lr=LEARNING_RATE, weight_decay=5e-4, lr_decay = 5e-4)
elif OPTIM == "adadelta":
    optimizer = torch.optim.Adadelta(gcn.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)
elif OPTIM == "radam":
    optimizer = torch.optim.RAdam(gcn.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)
elif OPTIM == "nadam":
    optimizer = torch.optim.NAdam(gcn.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)
elif OPTIM == "adamw":
    optimizer = torch.optim.AdamW(gcn.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)
elif OPTIM == "sparseadam":
    optimizer = torch.optim.SparseAdam(gcn.parameters(), lr=LEARNING_RATE)


epochs = EPOCHS

for t in range(epochs):
    train_loop(train_generator, gcn, loss_fn, optimizer)
    loss_train, correct_train = valid_test_loop(valid_generator, gcn, loss_fn)
    loss_test, correct_test = valid_test_loop(test_generator, gcn, loss_fn)
    avg_acc_train.append(correct_train)
    avg_acc_test.append(correct_test)
    loss_trains.append(loss_train)
    loss_tests.append(loss_test)


DATA["model"] = str(gcn)
DATA["optimizer"] = OPTIM
DATA["loss"] = LOSS
DATA["avg_acc_train"] = avg_acc_train
DATA["avg_acc_test"] = avg_acc_test
DATA["loss_tests"] = loss_tests
DATA["loss_trains"] = loss_trains

filehandler = open(f"{clusterPath}/DL_project/trained_model_{LEARNING_RATE}_{EPOCHS}_{BATCH_SIZE}_{CHEB_CHANNELS}_{CHEB_K}_{FC_NEURONS}_{OPTIM}_{LOSS}.pkl","wb")
pickle.dump(DATA,filehandler)
filehandler.close()