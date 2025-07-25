import cv2
import numpy as np


class Perception:
    """Class to handle perception tasks for the robot."""
    
    def __init__(self):
        """Initialize the perception module."""
        # Define the HSV range for red
        # Red wraps around the 0/360-degree hue boundary, so we need two ranges
        self.lower_red_1 = np.array([0, 120, 70])
        self.upper_red_1 = np.array([10, 255, 255])
        self.lower_red_2 = np.array([170, 120, 70])
        self.upper_red_2 = np.array([180, 255, 255])
    
    def rgb_to_hsv(self, rgb_image):
        """
        Convert an RGB image to HSV.
        
        Args:
            rgb_image (numpy.ndarray): RGB image.
            
        Returns:
            numpy.ndarray: HSV image.
        """
        # Ensure the image is in the correct format (uint8)
        if rgb_image.dtype != np.uint8:
            rgb_image = np.clip(rgb_image, 0, 255).astype(np.uint8)
        
        # Convert from RGB to BGR (OpenCV uses BGR)
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        
        # Convert from BGR to HSV
        hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        
        return hsv_image
    
    def detect_red(self, rgb_image):
        """
        Detect red objects in an RGB image.
        
        Args:
            rgb_image (numpy.ndarray): RGB image.
            
        Returns:
            numpy.ndarray: Binary mask of red objects.
        """
        try:
            # Convert to HSV
            hsv_image = self.rgb_to_hsv(rgb_image)
            
            # Create masks for the two red ranges
            mask1 = cv2.inRange(hsv_image, self.lower_red_1, self.upper_red_1)
            mask2 = cv2.inRange(hsv_image, self.lower_red_2, self.upper_red_2)
            
            # Combine the masks
            mask = cv2.bitwise_or(mask1, mask2)
            
            # Apply morphological operations to remove noise
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1)
            mask = cv2.dilate(mask, kernel, iterations=1)
            
            return mask
        except cv2.error:
            # If there's an error, return an empty mask of the right shape
            h, w = rgb_image.shape[:2]
            return np.zeros((h, w), dtype=np.uint8)
    
    def find_largest_contour(self, mask):
        """
        Find the largest contour in a binary mask.
        
        Args:
            mask (numpy.ndarray): Binary mask.
            
        Returns:
            tuple: (contour, area) of the largest contour, or (None, 0) if no contours found.
        """
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # If no contours found, return None
        if not contours:
            return None, 0
        
        # Find the largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        return largest_contour, area
    
    def calculate_centroid(self, contour):
        """
        Calculate the centroid of a contour.
        
        Args:
            contour (numpy.ndarray): Contour.
            
        Returns:
            tuple: (x, y) coordinates of the centroid, or None if the contour is invalid.
        """
        # Check if the contour is valid
        if contour is None:
            return None
        
        # Calculate the moments of the contour
        M = cv2.moments(contour)
        
        # Check if the contour has a non-zero area
        if M["m00"] == 0:
            return None
        
        # Calculate the centroid
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        
        return (cX, cY)
    
    def process_image(self, rgb_image):
        """
        Process an RGB image to detect red objects.
        
        Args:
            rgb_image (numpy.ndarray): RGB image.
            
        Returns:
            tuple: ((cX, cY), area) of the largest red object, or (None, 0) if no red objects found.
        """
        # Ensure the image is in the correct format
        if rgb_image.dtype != np.uint8:
            rgb_image = np.clip(rgb_image, 0, 255).astype(np.uint8)
            
        # Detect red objects
        mask = self.detect_red(rgb_image)
        
        # Find the largest contour
        contour, area = self.find_largest_contour(mask)
        
        # Calculate the centroid
        centroid = self.calculate_centroid(contour)
        
        return (centroid, area) if centroid else (None, 0)
    
    def visualize_detection(self, rgb_image, centroid=None, area=0):
        """
        Visualize the detection results.
        
        Args:
            rgb_image (numpy.ndarray): RGB image.
            centroid (tuple): (x, y) coordinates of the centroid.
            area (float): Area of the detected object.
            
        Returns:
            numpy.ndarray: Visualization image.
        """
        # Ensure the image is in the correct format
        if rgb_image.dtype != np.uint8:
            rgb_image = np.clip(rgb_image, 0, 255).astype(np.uint8)
            
        # Convert from RGB to BGR (OpenCV uses BGR)
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        
        # Create a copy of the image for visualization
        vis_image = bgr_image.copy()
        
        # Draw the mask
        mask = self.detect_red(rgb_image)
        red_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        red_mask[:, :, 0] = 0
        red_mask[:, :, 1] = 0
        
        # Blend the mask with the original image
        alpha = 0.3
        vis_image = cv2.addWeighted(vis_image, 1.0, red_mask, alpha, 0)
        
        # Draw the centroid if available
        if centroid:
            cv2.circle(vis_image, centroid, 5, (0, 255, 0), -1)
            cv2.putText(vis_image, f"Area: {area:.2f}", (centroid[0] + 10, centroid[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Convert back to RGB
        rgb_vis_image = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
        
        return rgb_vis_image 