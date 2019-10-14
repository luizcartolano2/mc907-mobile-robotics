import numpy as np
import cv2

def vrep2array(image, resolution):
    """
        Converts vrep image to a numpy BGR array.

    Args:
        image: List with raw pixel data received from vrep image sensor.
        resolution: Tuple with desired image resolution to create numpy array.

    Returns:
        img: Image as a numpy array in BGR.

    """
    img = np.array(image).astype(np.uint8)
    img = np.reshape(img, (resolution[1], resolution[0], -1))
    img = cv2.flip(img, 0) # Flip image over x-axis
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img
