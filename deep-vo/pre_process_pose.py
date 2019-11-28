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

def add_pi_to_poses(pose):
    '''Add Pi to every pose angle.'''
    pose += np.pi

    return pose

def quat_to_euler_angles(quat_matrix):
    # create a scipy object from the quaternion angles
    rot_mat = R.from_quat(quat_matrix)
    # convert the quaternion to euler (in degrees)
    euler_mat = rot_mat.as_euler('yxz', degrees=False)
    # convert from (-pi,pi) to (0,2pi)
    euler_convert = add_pi_to_poses(euler_mat)

    return euler_convert
