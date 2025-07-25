# Autonomous Mobile Manipulator

## Overview
This project implements a simulated mobile robot capable of detecting, approaching, and interacting with objects in its environment. The robot uses computer vision to identify red cubes, navigates to them, and interacts using a two-joint arm.

## Features
- Physics-based simulation using PyBullet
- Computer vision for object detection
- State machine for robot behavior control
- Two-wheeled differential drive robot with a manipulator arm
- Test-driven development approach

## Dependencies
See `requirements.txt` for a complete list of dependencies.

Main dependencies:
- PyBullet (physics simulation)
- OpenCV (computer vision)
- NumPy (numerical operations)
- SciPy (trajectory generation)
- Pytest (testing)
- Imageio (video recording)

## How to Run
1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the simulation: `python src/main.py`
4. Run tests: `pytest tests/`

## Code Structure
- `src/`: Source code
  - `main.py`: Main simulation loop
  - `environment.py`: Simulation environment setup
  - `robot.py`: Robot definition and control
  - `perception.py`: Computer vision and object detection
  - `planning.py`: Path planning and state machine
  - `utils.py`: Utility functions
- `tests/`: Test files
- `urdf/`: Robot and object model files
- `assets/`: Additional resources

## Configuration
Key parameters that can be tuned:
- `Kp`: Proportional gain for steering control
- `FORWARD_SPEED`: Base forward speed of the robot
- `STOPPING_AREA_THRESHOLD`: Area threshold for stopping near objects
- `HSV_RED_RANGES`: HSV color ranges for red object detection # MobileManipulator
