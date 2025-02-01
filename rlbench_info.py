import numpy as np
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.tasks import ReachTarget, CloseJarPeract

action_mode = MoveArmThenGripper(
    arm_action_mode=JointVelocity(),
    gripper_action_mode=Discrete()
)
env = Environment(action_mode)
env.launch()

task = env.get_task(CloseJarPeract)
descriptions, obs = task.reset()
# workspace_bounds = task.get_base_bounds()
# print("Workspace bounds (min, max):", workspace_bounds)
for i in range(500):
    obs, reward, terminate = task.step(np.random.normal(size=env.action_shape))
