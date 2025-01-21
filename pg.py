
from zero.v2.models.lotus.utils.rotation_transform import (
    RotationMatrixTransform, quaternion_to_discrete_euler
)
rot = [0, 36, 6]
gt_rot = quaternion_to_discrete_euler(rot, 5)
print(gt_rot)
