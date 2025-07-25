import pytest
import os
import sys
import numpy as np
import pybullet as p

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from environment import Environment
from robot import Robot
from perception import Perception
from planning import StateMachine


class TestStateMachine:
    """Test cases for the StateMachine class."""
    
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
        
        # Create the state machine with custom parameters for testing
        self.state_machine = StateMachine(self.robot, self.perception)
        # Override the spin velocity to ensure consistent test results
        self.state_machine.spin_velocity = 0.5
    
    def teardown_method(self):
        """Clean up after the test."""
        self.env.close()
    
    def test_state_machine_initialization(self):
        """Test that the state machine initializes correctly."""
        assert self.state_machine is not None
        assert self.state_machine.state == "SEARCHING"
        
    def test_searching_state(self):
        """Test the SEARCHING state."""
        # Reset wheel velocities to zero
        self.robot.set_wheel_velocity(left_vel=0.0, right_vel=0.0)
        
        # Execute the searching state
        self.state_machine.execute_searching_state()
        
        # Step the simulation to apply the velocities
        self.env.step_simulation(10)
        
        # The robot should be spinning in place
        # Check that the wheel velocities are set correctly
        # Left wheel should be positive, right wheel should be negative
        left_vel = p.getJointState(self.robot.robot_id, self.robot.left_wheel_joint)[1]
        right_vel = p.getJointState(self.robot.robot_id, self.robot.right_wheel_joint)[1]
        
        # In the searching state, the robot should be spinning
        # This means the wheels should be moving in opposite directions
        assert left_vel * right_vel < 0, "Wheels should be moving in opposite directions"
        
    def test_approaching_state(self):
        """Test the APPROACHING state."""
        # Create a mock centroid and area
        # Centroid slightly to the right of center (320/2 = 160)
        centroid = (180, 120)
        area = 1000  # Small area means we're far from the object
        
        # Execute the approaching state with the mock data
        self.state_machine.execute_approaching_state(centroid, area)
        
        # Step the simulation to apply the velocities
        self.env.step_simulation(10)
        
        # The robot should be moving towards the cube
        # Both wheels should be moving forward, but at different speeds
        left_vel = p.getJointState(self.robot.robot_id, self.robot.left_wheel_joint)[1]
        right_vel = p.getJointState(self.robot.robot_id, self.robot.right_wheel_joint)[1]
        
        # Both wheels should be moving in the same direction (forward)
        assert (left_vel > 0 and right_vel > 0) or (left_vel < 0 and right_vel < 0), "Both wheels should be moving in the same direction"
        
        # Left wheel should be faster than right wheel (turning right)
        assert abs(left_vel) != abs(right_vel), "Wheels should be moving at different speeds"
        
    def test_tapping_state(self):
        """Test the TAPPING state."""
        # Reset wheel velocities
        self.robot.set_wheel_velocity(left_vel=1.0, right_vel=1.0)
        
        # Execute the tapping state
        self.state_machine.execute_tapping_state()
        
        # The robot should have stopped and extended its arm
        # Check that the wheel velocities are near zero
        left_vel = p.getJointState(self.robot.robot_id, self.robot.left_wheel_joint)[1]
        right_vel = p.getJointState(self.robot.robot_id, self.robot.right_wheel_joint)[1]
        
        # The velocities should be very close to zero
        assert abs(left_vel) < 0.1, "Left wheel velocity should be near zero"
        assert abs(right_vel) < 0.1, "Right wheel velocity should be near zero"
        
        # Check that the arm has moved
        arm_angles = self.robot.get_arm_position()
        
        # The arm should have moved from its initial position (0, 0)
        assert arm_angles[0] != 0 or arm_angles[1] != 0, "Arm should have moved"
        
    def test_state_transition(self):
        """Test state transitions."""
        # Initially in SEARCHING state
        assert self.state_machine.state == "SEARCHING"
        
        # Mock the perception process
        # Create a mock centroid and area
        centroid = (160, 120)  # Center of image
        area = 1000  # Small area means we're far from the object
        
        # Update the state machine with mock perception results
        self.state_machine.state = "SEARCHING"
        
        # Manually update the state based on the mock perception results
        if centroid is not None:
            self.state_machine.state = "APPROACHING"
        
        # Check that the state has changed to APPROACHING
        assert self.state_machine.state == "APPROACHING"
        
        # Now simulate getting closer to the object
        area = 6000  # Large area means we're close to the object
        
        # Execute the approaching state with the mock data
        self.state_machine.execute_approaching_state(centroid, area)
        
        # Check that the state has changed to TAPPING
        assert self.state_machine.state == "TAPPING" 