import pytest
import os
import sys
import numpy as np
import cv2

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from environment import Environment
from robot import Robot
from perception import Perception


class TestPerception:
    """Test cases for the Perception class."""
    
    def setup_method(self):
        """Set up the test environment."""
        self.env = Environment(gui=False)
        self.robot_id = self.env.load_robot()
        self.robot = Robot(self.robot_id, self.env)
        self.perception = Perception()
        
        # Create a red cube in front of the robot
        self.cube_id = self.env.create_cube(
            position=[1.0, 0.0, 0.05],
            size=0.1,
            color=[1, 0, 0, 1]
        )
        
        # Step the simulation to settle
        self.env.step_simulation(10)
    
    def teardown_method(self):
        """Clean up after the test."""
        self.env.close()
    
    def test_perception_initialization(self):
        """Test that the perception module initializes correctly."""
        assert self.perception is not None
        
    def test_hsv_conversion(self):
        """Test that we can convert RGB to HSV."""
        # Get an image from the robot's camera
        rgb, _, _ = self.robot.get_camera_image()
        
        # Convert to HSV
        hsv = self.perception.rgb_to_hsv(rgb)
        
        # Check that the HSV image has the expected shape
        assert hsv.shape == rgb.shape
        
    def test_red_detection(self):
        """Test that we can detect red objects."""
        # Get an image from the robot's camera
        rgb, _, _ = self.robot.get_camera_image()
        
        # Detect red objects
        mask = self.perception.detect_red(rgb)
        
        # Check that the mask has the expected shape
        assert mask.shape == (rgb.shape[0], rgb.shape[1])
        
        # There should be some red pixels in the mask (the cube)
        assert np.sum(mask) > 0
        
    def test_find_largest_contour(self):
        """Test that we can find the largest contour in a mask."""
        # Get an image from the robot's camera
        rgb, _, _ = self.robot.get_camera_image()
        
        # Detect red objects
        mask = self.perception.detect_red(rgb)
        
        # Find the largest contour
        contour, area = self.perception.find_largest_contour(mask)
        
        # Check that a contour was found
        assert contour is not None
        assert area > 0
        
    def test_calculate_centroid(self):
        """Test that we can calculate the centroid of a contour."""
        # Get an image from the robot's camera
        rgb, _, _ = self.robot.get_camera_image()
        
        # Detect red objects
        mask = self.perception.detect_red(rgb)
        
        # Find the largest contour
        contour, _ = self.perception.find_largest_contour(mask)
        
        # Calculate the centroid
        centroid = self.perception.calculate_centroid(contour)
        
        # Check that a centroid was found
        assert centroid is not None
        assert len(centroid) == 2
        
    def test_process_image(self):
        """Test the complete image processing pipeline."""
        # Get an image from the robot's camera
        rgb, _, _ = self.robot.get_camera_image()
        
        # Process the image
        result = self.perception.process_image(rgb)
        
        # Check that a result was returned
        assert result is not None
        
        # The result should have a centroid and an area
        centroid, area = result
        
        # Check that a centroid was found
        assert centroid is not None
        assert len(centroid) == 2
        
        # Check that the area is positive
        assert area > 0 