import numpy as np
import time
import pybullet as p


class StateMachine:
    """Class to handle the robot's state machine."""
    
    def __init__(self, robot, perception):
        """
        Initialize the state machine.
        
        Args:
            robot (Robot): The robot object.
            perception (Perception): The perception object.
        """
        self.robot = robot
        self.perception = perception
        
        # Initialize the state
        self.state = "SEARCHING"
        
        # Parameters for the state machine
        self.spin_velocity = 0.5  # Velocity for spinning in place
        self.forward_speed = 1.0  # Base forward speed
        self.Kp = 0.01  # Proportional gain for steering
        self.stopping_area_threshold = 5000  # Area threshold for stopping
        
        # Parameters for the arm
        self.home_position = [0.0, 0.0]  # Home position for the arm
        self.extended_position = [0.8, -0.5]  # Extended position for tapping
        
        # Initialize the arm to the home position
        self.robot.set_arm_position(self.home_position)
    
    def execute_searching_state(self):
        """Execute the SEARCHING state."""
        # Spin in place to search for objects
        self.robot.set_wheel_velocity(
            left_vel=self.spin_velocity,
            right_vel=-self.spin_velocity
        )
    
    def execute_approaching_state(self, centroid, area):
        """
        Execute the APPROACHING state.
        
        Args:
            centroid (tuple): (x, y) coordinates of the centroid.
            area (float): Area of the detected object.
        """
        # Check if we have a valid centroid
        if centroid is None:
            # If no centroid, go back to searching
            self.state = "SEARCHING"
            return
        
        # Get the image center (assuming the image is 320x240)
        image_center_x = 320 // 2
        
        # Calculate the steering error
        error = image_center_x - centroid[0]
        
        # Use a P-controller to set wheel velocities
        right_velocity = self.forward_speed - self.Kp * error
        left_velocity = self.forward_speed + self.Kp * error
        
        # Set the wheel velocities
        self.robot.set_wheel_velocity(
            left_vel=left_velocity,
            right_vel=right_velocity
        )
        
        # Check if we're close enough to the object
        if area > self.stopping_area_threshold:
            # If close enough, transition to TAPPING
            self.state = "TAPPING"
    
    def execute_tapping_state(self):
        """Execute the TAPPING state."""
        # Stop the robot
        self.robot.set_wheel_velocity(left_vel=0, right_vel=0)
        
        # Execute the arm motion sequence
        self.robot.execute_arm_trajectory(
            target_angles=self.extended_position,
            duration=1.0,
            steps=50
        )
        
        # Wait a moment at the extended position
        time.sleep(0.5)
        
        # Return to the home position
        self.robot.execute_arm_trajectory(
            target_angles=self.home_position,
            duration=1.0,
            steps=50
        )
        
        # Transition to FINISHED
        self.state = "FINISHED"
    
    def execute_finished_state(self):
        """Execute the FINISHED state."""
        # Stop the robot
        self.robot.set_wheel_velocity(left_vel=0, right_vel=0)
        
        # Keep the arm in the home position
        self.robot.set_arm_position(self.home_position)
    
    def update(self, rgb_image):
        """
        Update the state machine based on the current image.
        
        Args:
            rgb_image (numpy.ndarray): RGB image from the robot's camera.
        """
        # Process the image to get the centroid and area
        centroid, area = self.perception.process_image(rgb_image)
        
        # Execute the current state
        if self.state == "SEARCHING":
            self.execute_searching_state()
            
            # If a centroid is found, transition to APPROACHING
            if centroid is not None:
                self.state = "APPROACHING"
                
        elif self.state == "APPROACHING":
            self.execute_approaching_state(centroid, area)
            
        elif self.state == "TAPPING":
            self.execute_tapping_state()
            
        elif self.state == "FINISHED":
            self.execute_finished_state()


class TSPPlanner:
    """Class to handle the TSP path planning."""
    
    def __init__(self, robot, perception):
        """
        Initialize the TSP planner.
        
        Args:
            robot (Robot): The robot object.
            perception (Perception): The perception object.
        """
        self.robot = robot
        self.perception = perception
        
        # List to store cube positions
        self.cube_positions = []
        
    def scan_environment(self, env, num_steps=360):
        """
        Scan the environment by rotating 360 degrees and detecting cubes.
        
        Args:
            env (Environment): The environment object.
            num_steps (int): Number of steps for the full rotation.
        """
        # Clear the cube positions
        self.cube_positions = []
        
        # Set the robot to spin in place
        self.robot.set_wheel_velocity(left_vel=0.2, right_vel=-0.2)
        
        # Scan the environment
        for _ in range(num_steps):
            # Step the simulation
            env.step_simulation(1)
            
            # Get an image from the robot's camera
            rgb, depth, _ = self.robot.get_camera_image()
            
            # Process the image to get the centroid and area
            centroid, area = self.perception.process_image(rgb)
            
            # If a cube is detected
            if centroid is not None:
                # Get the robot's position
                robot_pos, robot_orn = env.get_object_pose(self.robot.robot_id)
                
                # Calculate the world coordinates of the cube
                # This is a simplified calculation and would need to be refined
                # in a real implementation
                # Here we just use the robot's position plus an offset
                # based on the centroid position in the image
                image_center_x = 320 // 2
                image_center_y = 240 // 2
                
                # Calculate the offset from the image center
                offset_x = (centroid[0] - image_center_x) / image_center_x
                offset_y = (centroid[1] - image_center_y) / image_center_y
                
                # Convert the orientation quaternion to Euler angles
                euler = p.getEulerFromQuaternion(robot_orn)
                yaw = euler[2]
                
                # Calculate the cube position
                # This is a very simplified calculation
                # In a real implementation, you would use the depth information
                # and the camera's intrinsic and extrinsic parameters
                cube_x = robot_pos[0] + 1.0 * np.cos(yaw + offset_x)
                cube_y = robot_pos[1] + 1.0 * np.sin(yaw + offset_y)
                cube_z = 0.05  # Assuming the cube is on the ground
                
                cube_pos = [cube_x, cube_y, cube_z]
                
                # Check if this cube is already in our list
                # If not, add it
                is_new_cube = True
                for pos in self.cube_positions:
                    # If the cube is close to an existing cube, it's not new
                    if np.linalg.norm(np.array(pos) - np.array(cube_pos)) < 0.5:
                        is_new_cube = False
                        break
                
                if is_new_cube:
                    self.cube_positions.append(cube_pos)
        
        # Stop the robot
        self.robot.set_wheel_velocity(left_vel=0, right_vel=0)
    
    def generate_tsp_path(self, start_pos):
        """
        Generate a TSP path using the nearest neighbor algorithm.
        
        Args:
            start_pos (list): Starting position [x, y, z].
            
        Returns:
            list: Ordered list of positions to visit.
        """
        # If no cubes were found, return an empty path
        if not self.cube_positions:
            return []
        
        # Create a list of unvisited cubes
        unvisited = self.cube_positions.copy()
        
        # Start at the robot's position
        current_pos = start_pos
        path = []
        
        # While there are unvisited cubes
        while unvisited:
            # Find the closest unvisited cube
            closest_idx = 0
            closest_dist = float('inf')
            
            for i, pos in enumerate(unvisited):
                dist = np.linalg.norm(np.array(current_pos) - np.array(pos))
                if dist < closest_dist:
                    closest_dist = dist
                    closest_idx = i
            
            # Add the closest cube to the path
            closest_pos = unvisited.pop(closest_idx)
            path.append(closest_pos)
            
            # Update the current position
            current_pos = closest_pos
        
        return path 