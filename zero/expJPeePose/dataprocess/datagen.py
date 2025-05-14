"""
Generate the eePose and JP pairs
"""
from zero.z_utils.codebase.Rotation import *
import math
from zero.expForwardKinematics.ReconLoss.FrankaPandaFK import FrankaEmikaPanda
from zero.z_utils.coding import npa


frank = FrankaEmikaPanda()


JOINT_POSITION_LIMITS = npa([[-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
                             [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]])


low = JOINT_POSITION_LIMITS[0]
high = JOINT_POSITION_LIMITS[1]


# single sample
theta_sample = np.random.uniform(low, high)
print(theta_sample)  # shape (8,)

# batch of N samples
N = 16
batch = np.random.uniform(low, high, size=(N, low.shape[0]))
print(batch)   # shape (16, 8)
