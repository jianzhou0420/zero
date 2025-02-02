from rlbench.tasks import ReachTarget, CloseJarPeract, InsertOntoSquarePegPeract
from pyrep.objects.object import Object
from rlbench.tasks import ReachTarget  # Replace with your desired task
from rlbench.action_modes.action_mode import ActionMode, ArmActionMode, GripperActionMode, MoveArmThenGripper
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


# Initialize the RLBench environment
action_mode = MoveArmThenGripper(
    arm_action_mode=JointVelocity(),
    gripper_action_mode=Discrete()
)
env = Environment(action_mode, headless=True)
env.launch()

task = env.get_task(InsertOntoSquarePegPeract)
# Select and load a specific task
task = env.get_task(ReachTarget)  # Change to another task if needed
descriptions, obs = task.reset()  # Reset to get the initial observation

# Get all objects in the task scene
shape = task._scene._workspace

print(f"Name: {shape.get_name()}")
print(f"Type: {shape.get_type()}")
print(f"Size (Bounding Box): {shape.get_bounding_box()}")
print(f"Position: {shape.get_position()}")
print(f"Orientation (Euler): {shape.get_orientation()}")
print(f"Parent: {shape.get_parent()}")
print(f"Velocity: {shape.get_velocity()}")
print(f"Mass: {shape.get_mass()}")

print(f"Respondable: {shape.is_respondable()}")
print(f"Renderable: {shape.is_renderable()}")
print(f"Color (RGB): {shape.get_color()}")
print(f"Bounding Box: {shape.get_bounding_box()}")
print(f"Collidable: {shape.is_collidable()}")

print("-" * 50)
# Shutdown the environment
env.shutdown()
