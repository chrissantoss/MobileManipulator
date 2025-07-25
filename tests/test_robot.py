import pytest
import os
import sys
import numpy as np

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from environment import Environment
from robot import Robot


class TestRobot:
    """Test cases for the Robot class."""
    
    def setup_method(self):
        """Set up the test environment."""
        self.env = Environment(gui=False)
        self.robot_id = self.env.load_robot()
        self.robot = Robot(self.robot_id, self.env)
    
    def teardown_method(self):
        """Clean up after the test."""
        self.env.close()
    
    def test_robot_initialization(self):
        """Test that the robot initializes correctly."""
        assert self.robot.robot_id == self.robot_id
        assert self.robot.env == self.env
        assert len(self.robot.joint_indices) > 0
        
    def test_get_joint_info(self):
        """Test that we can get joint information."""
        for joint_index in self.robot.joint_indices:
            joint_info = self.robot.get_joint_info(joint_index)
            assert joint_info is not None
            
    def test_set_wheel_velocity(self):
        """Test that we can set wheel velocities."""
        # Get the initial position
        initial_pos, _ = self.env.get_object_pose(self.robot_id)
        
        # Set wheel velocities to move forward
        self.robot.set_wheel_velocity(left_vel=1.0, right_vel=1.0)
        
        # Step the simulation to apply the velocities
        self.env.step_simulation(100)
        
        # Get the new position
        new_pos, _ = self.env.get_object_pose(self.robot_id)
        
        # Check that the robot has moved (in any direction)
        # Calculate the Euclidean distance between initial and new positions
        distance = np.linalg.norm(np.array(new_pos) - np.array(initial_pos))
        
        # The robot should have moved some distance
        assert distance > 0.01, f"Robot did not move significantly. Distance: {distance}"
        
    def test_set_arm_position(self):
        """Test that we can set arm positions."""
        # Set the arm to a specific position
        target_angles = [0.5, -0.5]
        self.robot.set_arm_position(target_angles)
        
        # Step the simulation to apply the positions
        self.env.step_simulation(100)
        
        # Check that the arm joints are at the target positions
        current_angles = self.robot.get_arm_position()
        
        # Check that the angles are close to the target (with some tolerance)
        assert np.allclose(current_angles, target_angles, atol=0.1)
        
    def test_get_camera_image(self):
        """Test that we can get a camera image."""
        rgb, depth, seg = self.robot.get_camera_image()
        
        # Check that the images have the expected shape
        assert rgb.shape == (240, 320, 3)
        assert depth.shape == (240, 320)
        assert seg.shape == (240, 320) 