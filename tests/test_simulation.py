import os
import sys
import pytest
from unittest.mock import patch, MagicMock
import numpy as np

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from main import run_simulation


class MockPerception:
    """Mock Perception class for testing."""
    
    def process_image(self, rgb_image):
        """Mock image processing that always returns a fixed result."""
        return (None, 0)  # No object detected


@pytest.mark.skip(reason="Integration test that requires proper image processing")
def test_simulation_basic():
    """Test that the simulation runs without errors."""
    # Run the simulation without GUI and without recording
    run_simulation(gui=False, record=False, use_tsp=False)


@pytest.mark.skip(reason="Integration test that requires proper image processing")
def test_simulation_tsp():
    """Test that the simulation runs with TSP without errors."""
    # Run the simulation without GUI and without recording, but with TSP
    run_simulation(gui=False, record=False, use_tsp=True)


def test_simulation_with_mock():
    """Test the simulation with mocked perception."""
    with patch('perception.Perception', return_value=MockPerception()):
        # Run a very short simulation
        try:
            run_simulation(gui=False, record=False, use_tsp=False, max_steps=10)
            assert True  # If we get here without errors, the test passes
        except Exception as e:
            pytest.fail(f"Simulation failed with error: {e}")


def test_tsp_simulation_with_mock():
    """Test the TSP simulation with mocked perception."""
    with patch('perception.Perception', return_value=MockPerception()):
        # Run a very short simulation
        try:
            run_simulation(gui=False, record=False, use_tsp=True, max_steps=10)
            assert True  # If we get here without errors, the test passes
        except Exception as e:
            pytest.fail(f"TSP simulation failed with error: {e}") 