import os
import time
import numpy as np
import imageio
import argparse

from environment import Environment
from robot import Robot
from perception import Perception
from planning import StateMachine, TSPPlanner


def run_simulation(gui=True, record=True, use_tsp=False, max_steps=1000):
    """
    Run the simulation.
    
    Args:
        gui (bool): Whether to use the GUI.
        record (bool): Whether to record a video.
        use_tsp (bool): Whether to use TSP path planning.
        max_steps (int): Maximum number of simulation steps.
    """
    # Create the environment
    env = Environment(gui=gui)
    
    # Load the robot
    robot_id = env.load_robot(position=[0, 0, 0.1])
    robot = Robot(robot_id, env)
    
    # Create the perception module
    perception = Perception()
    
    # Create the state machine
    state_machine = StateMachine(robot, perception)
    
    # If using TSP, create the TSP planner
    if use_tsp:
        tsp_planner = TSPPlanner(robot, perception)
    
    # Create some red cubes
    if use_tsp:
        # Create multiple cubes for TSP
        cube_positions = [
            [1.0, 0.0, 0.05],
            [0.0, 1.0, 0.05],
            [-1.0, 0.0, 0.05],
            [0.0, -1.0, 0.05]
        ]
        
        for pos in cube_positions:
            env.create_cube(position=pos, size=0.1, color=[1, 0, 0, 1])
    else:
        # Create a single cube
        env.create_cube(position=[1.0, 0.0, 0.05], size=0.1, color=[1, 0, 0, 1])
    
    # Step the simulation to settle
    env.step_simulation(100)
    
    # List to store frames for video recording
    frames = []
    
    # If using TSP, first scan the environment
    if use_tsp:
        print("Scanning environment...")
        tsp_planner.scan_environment(env)
        
        # Get the robot's position
        robot_pos, _ = env.get_object_pose(robot_id)
        
        # Generate the TSP path
        path = tsp_planner.generate_tsp_path(robot_pos)
        
        print(f"Found {len(path)} cubes. Generating path...")
        
        # If no cubes were found, exit
        if not path:
            print("No cubes found. Exiting...")
            env.close()
            return
        
        # Set the first cube as the target
        current_target_idx = 0
        current_target = path[current_target_idx]
        
        print(f"Moving to cube {current_target_idx + 1}/{len(path)}")
    
    # Main simulation loop
    for step in range(max_steps):
        # Get an image from the robot's camera
        rgb, _, _ = robot.get_camera_image()
        
        # If recording, store the frame
        if record:
            frames.append(rgb)
        
        # Update the state machine
        state_machine.update(rgb)
        
        # If using TSP and we've finished with the current target
        if use_tsp and state_machine.state == "FINISHED":
            # Move to the next target
            current_target_idx += 1
            
            # If we've visited all targets, we're done
            if current_target_idx >= len(path):
                print("All cubes visited. Exiting...")
                break
            
            # Set the next target
            current_target = path[current_target_idx]
            
            # Reset the state machine to SEARCHING
            state_machine.state = "SEARCHING"
            
            print(f"Moving to cube {current_target_idx + 1}/{len(path)}")
        
        # If we've finished and we're not using TSP, we're done
        if not use_tsp and state_machine.state == "FINISHED":
            print("Finished. Exiting...")
            break
        
        # Step the simulation
        env.step_simulation(1)
        
        # Sleep to control the simulation speed
        if gui:
            time.sleep(0.01)
    
    # If recording, save the video
    if record and frames:
        print("Saving video...")
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../assets'))
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'robot_pov.mp4')
        
        # Save the video
        writer = imageio.get_writer(output_path, fps=20)
        for frame in frames:
            writer.append_data(frame)
        writer.close()
        
        print(f"Video saved to {output_path}")
    
    # Close the environment
    env.close()


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run the mobile robot simulation.")
    parser.add_argument("--gui", action="store_true", help="Use the GUI.")
    parser.add_argument("--record", action="store_true", help="Record a video.")
    parser.add_argument("--tsp", action="store_true", help="Use TSP path planning.")
    parser.add_argument("--max-steps", type=int, default=1000, help="Maximum number of simulation steps.")
    args = parser.parse_args()
    
    # Run the simulation
    run_simulation(gui=args.gui, record=args.record, use_tsp=args.tsp, max_steps=args.max_steps) 