import pybullet as p
import numpy as np
import time


class Robot:
    """Class to handle the robot in the simulation."""
    
    def __init__(self, robot_id, env):
        """
        Initialize the robot.
        
        Args:
            robot_id (int): ID of the robot in the simulation.
            env (Environment): The environment object.
        """
        self.robot_id = robot_id
        self.env = env
        
        # Get the number of joints in the robot
        self.num_joints = p.getNumJoints(self.robot_id)
        
        # Store the joint indices
        self.joint_indices = range(self.num_joints)
        
        # Find the wheel joint indices
        self.left_wheel_joint = None
        self.right_wheel_joint = None
        
        # Find the arm joint indices
        self.arm_joints = []
        
        # Identify the joints
        for i in range(self.num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            joint_name = joint_info[1].decode('utf-8')
            
            if joint_name == "left_wheel_joint":
                self.left_wheel_joint = i
            elif joint_name == "right_wheel_joint":
                self.right_wheel_joint = i
            elif joint_name == "arm_joint1" or joint_name == "arm_joint2":
                self.arm_joints.append(i)
        
        # Check that we found all the joints we need
        assert self.left_wheel_joint is not None, "Left wheel joint not found"
        assert self.right_wheel_joint is not None, "Right wheel joint not found"
        assert len(self.arm_joints) == 2, "Expected 2 arm joints, found {}".format(len(self.arm_joints))
        
        # Camera parameters
        self.camera_width = 320
        self.camera_height = 240
        self.camera_fov = 60
        self.camera_aspect = float(self.camera_width) / self.camera_height
        self.camera_near = 0.1
        self.camera_far = 100.0
        
    def get_joint_info(self, joint_index):
        """
        Get information about a joint.
        
        Args:
            joint_index (int): Index of the joint.
            
        Returns:
            tuple: Joint information.
        """
        return p.getJointInfo(self.robot_id, joint_index)
    
    def set_wheel_velocity(self, left_vel, right_vel):
        """
        Set the velocity of the wheels.
        
        Args:
            left_vel (float): Velocity of the left wheel in rad/s.
            right_vel (float): Velocity of the right wheel in rad/s.
        """
        p.setJointMotorControl2(
            bodyUniqueId=self.robot_id,
            jointIndex=self.left_wheel_joint,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=left_vel
        )
        
        p.setJointMotorControl2(
            bodyUniqueId=self.robot_id,
            jointIndex=self.right_wheel_joint,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=right_vel
        )
    
    def set_arm_position(self, target_angles):
        """
        Set the position of the arm joints.
        
        Args:
            target_angles (list): Target angles for the arm joints in radians.
        """
        assert len(target_angles) == len(self.arm_joints), "Number of target angles must match number of arm joints"
        
        for i, joint_index in enumerate(self.arm_joints):
            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=joint_index,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_angles[i]
            )
    
    def get_arm_position(self):
        """
        Get the current position of the arm joints.
        
        Returns:
            list: Current angles of the arm joints in radians.
        """
        arm_angles = []
        
        for joint_index in self.arm_joints:
            joint_state = p.getJointState(self.robot_id, joint_index)
            arm_angles.append(joint_state[0])
        
        return arm_angles
    
    def get_camera_image(self):
        """
        Get an image from the robot's camera.
        
        Returns:
            tuple: (rgb_array, depth_array, segmentation_array) from the camera.
        """
        # Get the position and orientation of the robot
        pos, orn = self.env.get_object_pose(self.robot_id)
        
        # Calculate the camera position (slightly above and in front of the robot)
        camera_pos = [pos[0], pos[1], pos[2] + 0.2]
        
        # Calculate the target position (in front of the robot)
        # Convert the orientation quaternion to Euler angles
        euler = p.getEulerFromQuaternion(orn)
        yaw = euler[2]
        
        # Calculate a point in front of the robot
        target_x = pos[0] + 1.0 * np.cos(yaw)
        target_y = pos[1] + 1.0 * np.sin(yaw)
        target_z = pos[2]
        
        target_pos = [target_x, target_y, target_z]
        
        # Calculate the up vector
        up_vector = [0, 0, 1]
        
        # Compute the view matrix
        view_matrix = self.env.get_camera_view_matrix(
            camera_pos, target_pos, up_vector
        )
        
        # Compute the projection matrix
        proj_matrix = self.env.get_camera_projection_matrix(
            fov=self.camera_fov,
            aspect=self.camera_aspect,
            near=self.camera_near,
            far=self.camera_far
        )
        
        # Get the camera image
        return self.env.get_camera_image(
            view_matrix, proj_matrix,
            width=self.camera_width,
            height=self.camera_height
        )
    
    def execute_arm_trajectory(self, target_angles, duration=1.0, steps=50):
        """
        Execute a smooth trajectory for the arm.
        
        Args:
            target_angles (list): Target angles for the arm joints in radians.
            duration (float): Duration of the trajectory in seconds.
            steps (int): Number of steps in the trajectory.
        """
        # Get the current arm position
        start_angles = self.get_arm_position()
        
        # Generate a simple linear trajectory
        for t in range(steps + 1):
            alpha = t / steps
            
            # Linear interpolation between start and target angles
            current_angles = [
                start_angles[i] + alpha * (target_angles[i] - start_angles[i])
                for i in range(len(start_angles))
            ]
            
            # Set the arm position
            self.set_arm_position(current_angles)
            
            # Step the simulation
            self.env.step_simulation(1)
            
            # Sleep to control the execution speed
            time.sleep(duration / steps) 