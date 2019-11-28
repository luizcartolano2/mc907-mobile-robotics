import pandas as pd
import glob
import numpy as np
import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torch.autograd import Function
from torch.autograd import Variable
from torchvision import models
import torch.optim as optim
import math
from scipy.spatial.transform import Rotation as R
import matplotlib
import matplotlib.pyplot as plt
from load_data import *
from model import *
from train import *

from azureml.core import Run


X,y = VODataLoader(datapath='dataset', test=False)

X_train = [item for x in X for item in x]
Y_train = [item for a in y for item in a]

X_stack = torch.stack(X_train)
y_stack = torch.stack(Y_train)
X_batch = X_stack.view(-1,1,6,144,256)
y_batch = y_stack.view(-1,1,6)
validation_split = .2
dataset_size = len(X_batch)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))

# get hold of the current run
run = Run.get_context()

# creating model
modelVO = DeepVONet()
# defining loss and optimizer to be used
criterion = torch.nn.MSELoss()
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.5, weight_decay=0.5)
optimizer = optim.Adagrad(model.parameters(), lr=0.0005)

train_loss, valid_loss = training_model_v2(model, X_batch_train, y_batch_train, X_batch_validation, y_batch_validation, 60)

os.makedirs('outputs', exist_ok=True)
# note file saved in the outputs folder is automatically uploaded into experiment record
torch.save(model.state_dict(), 'outputs/deepvo-dropout.pt')
