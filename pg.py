import numpy as np

pos_bins = 15
pos_bin_size = 0.001

# Define scaling factor
scaling_factor = 1.8  # Increase for a steeper growth

# Generate nonlinear bin positions
bins = np.arange(-pos_bins, pos_bins)
shift = np.sign(bins) * (np.abs(bins) ** scaling_factor) * pos_bin_size

print(shift)
