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
from pre_process_image import *
from pre_process_pose import *

def load_images(img_dir, img_size=(256,144)):

    """
        Function to coordinate the load of all the images that are going to be used.

        :param: img_dir - path to the directory containing the images
        :param: img_size - image size

        :return: images_set - numpy array with all images at the set

    """
    print("----------------------------------------------------------------------")
    print ("|    Loading images from: ", img_dir)

    # loop to read all the images of the directory
    images = [get_image(img,img_size) for img in glob.glob(img_dir+'/*')]

    # normalize images
    images = mean_normalize(images)

    #resemble images as RGB
    images = [img[:, :, (2, 1, 0)] for img in images]

    #Transpose the image that channels num. as first dimension
    images = [np.transpose(img,(2,0,1)) for img in images]
    images = [torch.from_numpy(img) for img in images]

    #stack per 2 images
    images = [np.concatenate((images[k],images[k+1]),axis = 0) for k in range(len(images)-1)]

    print("|    Images count : ",len(images))

    # reshape the array of all images
    images_set = np.stack(images,axis=0)
    print("----------------------------------------------------------------------")

    return images_set

def load_airsim_pose(pose_file):

    poses = []
    poses_set = []

    df_poses = pd.read_csv(pose_file)
    for index, row in df_poses.iterrows():
        # get the (x,y,z) positions of the camera
        position = np.array([row['POS_X'],row['POS_Y'],row['POS_Z']])
        # get the quaternions angles of the camera
        quat_matrix = np.array([row['Q_X'],row['Q_Y'], row['Q_Z'],row['Q_W']])
        # call the func that convert the quaternions to euler angles
        euler_matrix = quat_to_euler_angles(quat_matrix)
        # concatenate both position(x,y,z) and euler angles
        poses.append(np.concatenate((position,euler_matrix)))

    # make the first pose as start position
    pose1 = poses[0]
    for i in range(len(poses)):
        pose2 = poses[i]
        pose_diff = np.subtract(pose2, pose1)
        pose_diff[4:] = np.arctan2(np.sin(pose_diff[4:]), np.cos(pose_diff[4:]))

        poses[i] = pose_diff

    # get the desloc between two poses
    for i in range(len(poses)-1):
        pose1 = poses[i]
        pose2 = poses[i+1]

        pose_diff = np.subtract(pose2, pose1)
        pose_diff[4:] = np.arctan2(np.sin(pose_diff[4:]), np.cos(pose_diff[4:]))

        poses_set.append(pose_diff)

    return poses_set

def load_poses(pose_file, pose_format='airsim'):
    """
        Function to load the image poses.

        :param: pose_file - path to the pose file
        :param: pose_format - where the pose were obtained from (AirSim, VREP, Kitti, etc...)

        :return: pose_set - set of the poses for the sequence
    """
    print("----------------------------------------------------------------------")
    print ("|    Pose from: ",pose_file)

    if pose_format.lower() == 'kitti':
        poses_set = load_kitti_images(pose_file)
    elif pose_format.lower() == 'airsim':
        poses_set = load_airsim_pose(pose_file)

    print("|        Poses count: ",len(poses_set))
    print("----------------------------------------------------------------------")
    return poses_set

def VODataLoader(datapath,img_size=(256,144), test=False):
    if test:
        sequences = ['3']
    else:
        sequences = ['1','2','4','5','6']

    images_set = []
    odometry_set = []

    for sequence in sequences:
        dir_path = os.path.join(datapath,'seq'+sequence)
        image_path = os.path.join(dir_path,'images')
        pose_path = os.path.join(dir_path,'poses.csv')
        print("-----------------------------------------------------------------------")
        print("|Load from: ", dir_path)
        images_set.append(torch.FloatTensor(load_images(image_path,img_size)))
        odometry_set.append(torch.FloatTensor(load_poses(pose_path, 'AirSim')))
        print("-----------------------------------------------------------------------")

    print("---------------------------------------------------")
    print("|   Total Images: ", len(images_set))
    print("|   Total Odometry: ", len(odometry_set))
    print("---------------------------------------------------")
    return images_set, odometry_set
