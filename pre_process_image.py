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

def compute_rgb_mean(image_sequence):
    '''
        Compute the mean over each channel separately over a set of images.
        Parameters
        ----------
        image_sequence  :   np.ndarray
                        Array of shape ``(N, h, w, c)`` or ``(h, w, c)``
    '''
    if image_sequence.ndim == 4:
        _, h, w, c = image_sequence.shape
    if image_sequence.ndim == 3:
        h, w, c = image_sequence.shape
    # compute mean separately for each channel
    # somehow this expression is buggy, so we must do it manually
    # mode = image_sequence.mean((0, 1, 2))
    mean_r = image_sequence[..., 0].mean()
    mean_g = image_sequence[..., 1].mean()
    mean_b = image_sequence[..., 2].mean()
    mean = np.array([mean_r, mean_g, mean_b])
    return mean

def mean_normalize(images_vector):
    '''
        Normalize data to the range -1 to 1
    '''
    out_images = []

    N = len(images_vector)

    print('|    Mean-normalizing ...')

    mean_accumlator = np.zeros((3,), dtype=np.float32)

    for idx in range(N):
        img = images_vector[idx]
        mean_accumlator += compute_rgb_mean(img)

    mean_accumlator /= N
    print(f'|    Mean: {mean_accumlator}')

    for idx in range(N):
        img = images_vector[idx]
        out_images.append(img - mean_accumlator)

    print('|    Done')

    return out_images

def get_image(path,img_size=(256,144)):
    """
        Function to read an image from a given path.

        :param: path - image path
        :param: img_size - image size

        :return: img - numpy array with the images pixels (converted to grayscale and normalized)

    """
    # read image from path
    img = cv2.imread(path)

    # normalize image pixels
    img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    return img
