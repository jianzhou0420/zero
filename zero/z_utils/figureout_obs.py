from scipy.spatial.transform import Rotation as R
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointPosition
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.tasks import MT15_V1, CloseJar
from rlbench.observation_config import ObservationConfig
import numpy as np

action_mode = MoveArmThenGripper(
    arm_action_mode=JointPosition(absolute_mode=True),
    gripper_action_mode=Discrete()
)

observation_config = ObservationConfig()
observation_config.gripper_joint_positions = True

env = Environment(action_mode, obs_config=observation_config, headless=True)
env.launch()
# task: a rlbench task object

task = env.get_task(CloseJar)

task.sample_variation()  # random variation
des, obs = task.reset()
text = str(des[np.random.randint(len(des))])
terminate = False

# max_time_steps = 180
# for t in range(max_time_steps):
#     action = np.random.rand(8)
#     obs, reward, terminate = task.step(action)
#     terminate = terminate.T
# success, _ = task._task.success()

task.sample_variation()  # random variation
des, obs = task.reset()
text = str(des[np.random.randint(len(des))])


print('end')
test = task._task.get_base().get_objects_in_tree()
test2 = task._task.get_base().get_pose(test[1])

# homogeneous_matrix
print(test)
'''
pose = [0.5, 0.3, 0.2, 0.0, 0.707, 0.0, 0.707]
x, y, z, qx, qy, qz, qw
obs.gripper_pose :[7] 
'''
