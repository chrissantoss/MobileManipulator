import numpy as np
import cv2
import matplotlib.pyplot as plt


def visualize_image(rgb_image, title="RGB Image"):
    """
    Visualize an RGB image.
    
    Args:
        rgb_image (numpy.ndarray): RGB image.
        title (str): Title for the plot.
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(rgb_image)
    plt.title(title)
    plt.axis('off')
    plt.show()


def visualize_mask(mask, title="Mask"):
    """
    Visualize a binary mask.
    
    Args:
        mask (numpy.ndarray): Binary mask.
        title (str): Title for the plot.
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(mask, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()


def visualize_detection(rgb_image, mask, centroid=None, area=0):
    """
    Visualize the detection results.
    
    Args:
        rgb_image (numpy.ndarray): RGB image.
        mask (numpy.ndarray): Binary mask.
        centroid (tuple): (x, y) coordinates of the centroid.
        area (float): Area of the detected object.
    """
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    
    # Plot the RGB image
    ax1.imshow(rgb_image)
    ax1.set_title("RGB Image")
    ax1.axis('off')
    
    # If a centroid is available, plot it
    if centroid:
        ax1.plot(centroid[0], centroid[1], 'go', markersize=10)
        ax1.text(centroid[0] + 10, centroid[1], f"Area: {area:.2f}", color='green', fontsize=12)
    
    # Plot the mask
    ax2.imshow(mask, cmap='gray')
    ax2.set_title("Mask")
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()


def calculate_distance(pos1, pos2):
    """
    Calculate the Euclidean distance between two positions.
    
    Args:
        pos1 (list): First position [x, y, z].
        pos2 (list): Second position [x, y, z].
        
    Returns:
        float: Euclidean distance.
    """
    return np.linalg.norm(np.array(pos1) - np.array(pos2))


def quaternion_to_euler(quaternion):
    """
    Convert a quaternion to Euler angles (roll, pitch, yaw).
    
    Args:
        quaternion (list): Quaternion [x, y, z, w].
        
    Returns:
        list: Euler angles [roll, pitch, yaw] in radians.
    """
    import pybullet as p
    return p.getEulerFromQuaternion(quaternion)


def euler_to_quaternion(euler):
    """
    Convert Euler angles to a quaternion.
    
    Args:
        euler (list): Euler angles [roll, pitch, yaw] in radians.
        
    Returns:
        list: Quaternion [x, y, z, w].
    """
    import pybullet as p
    return p.getQuaternionFromEuler(euler) 