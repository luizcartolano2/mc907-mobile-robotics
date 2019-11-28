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

def training_model_v2(model, X_train, y_train, X_validate, y_validate, num_epochs):
    # start model train mode
    train_ep_loss = []
    valid_ep_loss = []

    for ep in range(num_epochs):
        st_t = time.time()
        print('='*50)

        # Train
        model.train()

        loss_mean = 0
        t_loss_list = []

        for i in range(X_train.size(0)):
            # get the images inputs
            inputs = X_train[i]
            # get the original poses
            labels = y_train[i]
            # zero optimizer
            optimizer.zero_grad()
            model.reset_hidden_states()

            # predict outputs
            outputs = model(inputs)
            # get mse loss
            loss = criterion(outputs, labels)
            ls = loss.item()
            loss.backward()
            # set next optimizer step
            optimizer.step()
            # append loss
            t_loss_list.append(float(ls))
            # update loss
            loss_mean += float(ls)

        print('Train take {:.1f} sec'.format(time.time()-st_t))
        loss_mean /= (X_train.size(0))
        train_ep_loss.append(loss_mean)

        # Validation
        st_t = time.time()
        model.eval()
        loss_mean_valid = 0
        v_loss_list = []

        for i in range(X_validate.size(0)):
            # get the images inputs
            inputs = X_validate[i]
            # get the original poses
            labels = y_validate[i]
            # predict outputs
            outputs = model(inputs)
            # get mse loss
            loss = criterion(outputs, labels)
            ls = loss.item()
            # update loss values
            v_loss_list.append(float(ls))
            loss_mean_valid += float(ls)

        print('Valid take {:.1f} sec'.format(time.time()-st_t))
        loss_mean_valid /= X_validate.size(0)
        valid_ep_loss.append(loss_mean_valid)

        print('Epoch {}\ntrain loss mean: {}, std: {:.2f}\nvalid loss mean: {}, std: {:.2f}\n'.format(ep+1, loss_mean, np.std(t_loss_list), loss_mean_valid, np.std(v_loss_list)))

    return train_ep_loss, valid_ep_loss
