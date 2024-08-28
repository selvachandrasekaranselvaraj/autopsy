# Autopsy

Autopsy is a Python package for analyzing molecular dynamics trajectories, with a focus on computing properties such as mean-squared displacement (MSD) and non-Gaussian parameter (NGP).

## Installation

You can install Autopsy using pip:

pip install autopsy

## Usage
#To use Autopsy, you can follow these basic steps:

# Read Trajectory and Sort Atomic Indices:

from autopsy.util import read_trajectory, sort_atomic_indices
from autopsy import PropertyCalculator

# Read trajectory data and sort atomic indices
data = read_trajectory.read_trajectory()
sorted_indices = sort_atomic_indices(data.positions, data.cell)

# Calculate Properties using PropertyCalculator:

from autopsy import PropertyCalculator

# Initialize PropertyCalculator with trajectory data
calculator = PropertyCalculator(data)

# Run the calculations
calculator.run()


