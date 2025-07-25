import pybullet as p
import pybullet_data
import numpy as np
import os
import time


class Environment:
    """Class to handle the PyBullet simulation environment."""
    
    def __init__(self, gui=True):
        """
        Initialize the PyBullet physics simulation.
        
        Args:
            gui (bool): Whether to use the GUI (True) or direct mode (False).
        """
        # Connect to the physics server
        self.physics_client = p.connect(p.GUI if gui else p.DIRECT)
        
        # Set up the path to the PyBullet data
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Set gravity
        p.setGravity(0, 0, -9.8)
        
        # Load the ground plane
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Store the path to our URDF files
        self.urdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../urdf'))
    
    def create_cube(self, position, size=0.1, color=[1, 0, 0, 1]):
        """
        Create a cube in the simulation.
        
        Args:
            position (list): [x, y, z] position of the cube.
            size (float): Size of the cube.
            color (list): [r, g, b, a] color of the cube.
            
        Returns:
            int: ID of the created cube.
        """
        # Create visual and collision shapes for the cube
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[size/2, size/2, size/2],
            rgbaColor=color
        )
        
        collision_shape_id = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[size/2, size/2, size/2]
        )
        
        # Create the multibody
        cube_id = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=collision_shape_id,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=position
        )
        
        return cube_id
    
    def load_robot(self, position=[0, 0, 0.1]):
        """
        Load the robot URDF into the simulation.
        
        Args:
            position (list): [x, y, z] position to place the robot.
            
        Returns:
            int: ID of the loaded robot.
        """
        # Load the robot URDF
        robot_path = os.path.join(self.urdf_path, "robot.urdf")
        robot_id = p.loadURDF(robot_path, position)
        
        return robot_id
    
    def get_object_pose(self, object_id):
        """
        Get the position and orientation of an object.
        
        Args:
            object_id (int): ID of the object.
            
        Returns:
            tuple: (position, orientation) of the object.
        """
        return p.getBasePositionAndOrientation(object_id)
    
    def get_camera_view_matrix(self, camera_position, target_position, up_vector):
        """
        Get the view matrix for a camera.
        
        Args:
            camera_position (list): [x, y, z] position of the camera.
            target_position (list): [x, y, z] position the camera is looking at.
            up_vector (list): [x, y, z] up vector for the camera.
            
        Returns:
            numpy.ndarray: View matrix for the camera.
        """
        return np.array(p.computeViewMatrix(
            cameraEyePosition=camera_position,
            cameraTargetPosition=target_position,
            cameraUpVector=up_vector
        ))
    
    def get_camera_projection_matrix(self, fov=60.0, aspect=1.0, near=0.1, far=100.0):
        """
        Get the projection matrix for a camera.
        
        Args:
            fov (float): Field of view in degrees.
            aspect (float): Aspect ratio (width / height).
            near (float): Near clipping plane.
            far (float): Far clipping plane.
            
        Returns:
            numpy.ndarray: Projection matrix for the camera.
        """
        return np.array(p.computeProjectionMatrixFOV(
            fov=fov,
            aspect=aspect,
            nearVal=near,
            farVal=far
        ))
    
    def get_camera_image(self, view_matrix, proj_matrix, width=320, height=240):
        """
        Get an image from a camera.
        
        Args:
            view_matrix (numpy.ndarray): View matrix for the camera.
            proj_matrix (numpy.ndarray): Projection matrix for the camera.
            width (int): Width of the image in pixels.
            height (int): Height of the image in pixels.
            
        Returns:
            tuple: (rgb_array, depth_array, segmentation_array) from the camera.
        """
        # Get the camera image
        width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix
        )
        
        # Convert the RGB image to a numpy array
        rgb_array = np.array(rgb_img).reshape(height, width, 4)[:, :, :3]
        
        # Convert the depth image to a numpy array
        depth_array = np.array(depth_img).reshape(height, width)
        
        # Convert the segmentation image to a numpy array
        seg_array = np.array(seg_img).reshape(height, width)
        
        return rgb_array, depth_array, seg_array
    
    def step_simulation(self, num_steps=1):
        """
        Step the simulation forward.
        
        Args:
            num_steps (int): Number of steps to take.
        """
        for _ in range(num_steps):
            p.stepSimulation()
    
    def close(self):
        """Close the PyBullet connection."""
        p.disconnect(self.physics_client) 