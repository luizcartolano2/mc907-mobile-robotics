import sys, time, cv2
from matplotlib import pyplot as plt

sys.path.insert(0, '/Users/luizeduardocartolano/Dropbox/DUDU/Unicamp/IC/MC906/workspace/robot-fuzzy/vrep-robot-master/src')
from robot import Robot
from utils import *

def display_image(image):
    """
        Displays a image with matplotlib.
        Args:
            image: The BGR image numpy array. See src/utils.py.
    """
    plt.imshow(image, cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()

robot = Robot()

#Reading ultrassonic sensors
ultrassonic = robot.read_ultrassonic_sensors()
print("Ultrassonic: ", ultrassonic)

#Reading laser sensor
laser = robot.read_laser()
print("Laser: ", laser)

#Reading camera
resolution, raw_img = robot.read_vision_sensor()
img = vrep2array(raw_img, resolution)
display_image(img)
