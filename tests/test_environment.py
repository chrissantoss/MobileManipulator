import pytest
import os
import sys
import numpy as np

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from environment import Environment


class TestEnvironment:
    """Test cases for the Environment class."""
    
    def test_environment_initialization(self):
        """Test that the environment initializes correctly."""
        env = Environment(gui=False)
        assert env.physics_client is not None
        
    def test_ground_plane_loading(self):
        """Test that the ground plane is loaded correctly."""
        env = Environment(gui=False)
        # The plane ID should be valid
        assert env.plane_id is not None
        # In PyBullet, the first object loaded typically gets ID 0 or 1
        assert env.plane_id in [0, 1]
        
    def test_cube_creation(self):
        """Test that cubes can be created with the correct properties."""
        env = Environment(gui=False)
        cube_id = env.create_cube(position=[1, 1, 0.5], size=0.1, color=[1, 0, 0, 1])
        
        # Get the position of the cube
        pos, _ = env.get_object_pose(cube_id)
        
        # Check that the position is correct (with some tolerance for physics)
        assert np.allclose(pos, [1, 1, 0.5], atol=1e-2)
        
    def test_robot_loading(self):
        """Test that the robot is loaded correctly."""
        env = Environment(gui=False)
        robot_id = env.load_robot(position=[0, 0, 0.1])
        
        # Check that the robot ID is valid
        assert robot_id >= 0
        
        # Get the position of the robot
        pos, _ = env.get_object_pose(robot_id)
        
        # Check that the position is correct (with some tolerance for physics)
        assert np.allclose(pos, [0, 0, 0.1], atol=1e-2)
        
    def test_camera_setup(self):
        """Test that the camera is set up correctly."""
        env = Environment(gui=False)
        
        # Set up a camera
        view_matrix = env.get_camera_view_matrix([0, 0, 1], [1, 0, 0], [0, 0, 1])
        proj_matrix = env.get_camera_projection_matrix(fov=60, aspect=1.0, near=0.1, far=100)
        
        # Check that the matrices are valid
        assert view_matrix.shape == (16,)
        assert proj_matrix.shape == (16,)
        
    def test_step_simulation(self):
        """Test that the simulation can step forward."""
        env = Environment(gui=False)
        
        # Step the simulation
        env.step_simulation(num_steps=100)
        
        # No assertion needed, just checking that it doesn't crash 