
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import re
import numpy as np
import pickle
import rlbench
import h5py
from tqdm import tqdm
import multiprocessing
DEFAULT_RGB_SCALE_FACTOR = 2**24 - 1  # JIAN: note this is actually the scale factor that rlbench uses not its default 2560000
DEFAULT_GRAY_SCALE_FACTOR = {np.uint8: 100.0,
                             np.uint16: 1000.0,
                             np.int32: DEFAULT_RGB_SCALE_FACTOR}


def image_to_float_array(image, scale_factor=None):
    '''
    Jian: From RLBench: https://github.com/stepjam/RLBench/blob/master/rlbench/backend/utils.py
    seems it need the image to have shape [H,W,3]
    '''

    """Recovers the depth values from an image.

    Reverses the depth to image conversion performed by FloatArrayToRgbImage or
    FloatArrayToGrayImage.

    The image is treated as an array of fixed point depth values.  Each
    value is converted to float and scaled by the inverse of the factor
    that was used to generate the Image object from depth values.  If
    scale_factor is specified, it should be the same value that was
    specified in the original conversion.

    The result of this function should be equal to the original input
    within the precision of the conversion.

    Args:
    image: Depth image output of FloatArrayTo[Format]Image.
    scale_factor: Fixed point scale factor.

    Returns:
    A 2D floating point numpy array representing a depth image.
    """
    image_array = np.array(image)
    image_dtype = image_array.dtype
    image_shape = image_array.shape

    channels = image_shape[2] if len(image_shape) > 2 else 1
    assert 2 <= len(image_shape) <= 3
    if channels == 3:
        # RGB image needs to be converted to 24 bit integer.
        float_array = np.sum(image_array * [65536, 256, 1], axis=2)
        if scale_factor is None:
            scale_factor = DEFAULT_RGB_SCALE_FACTOR
    else:
        if scale_factor is None:
            scale_factor = DEFAULT_GRAY_SCALE_FACTOR[image_dtype.type]
        float_array = image_array.astype(np.float32)
    scaled_array = float_array / scale_factor
    return scaled_array


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


class JianRLBenchDataStructure:
    '''
    This class is used to contain one episode data
    '''

    def __init__(self):
        self.left_shoulder_depth = None
        self.left_shoulder_mask = None
        self.left_shoulder_rgb = None
        self.right_shoulder_depth = None
        self.right_shoulder_mask = None
        self.right_shoulder_rgb = None
        self.wrist_depth = None
        self.wrist_mask = None
        self.wrist_rgb = None
        self.front_depth = None
        self.front_mask = None
        self.front_rgb = None
        self.overhead_depth = None
        self.overhead_mask = None
        self.overhead_rgb = None

        self.variation_descriptions = None
        self.variation_number = None

        self.gripper_joint_positions = None
        self.gripper_matrix = None
        self.gripper_open = None
        self.gripper_touch_forces = None

        self.joint_forces = None
        self.joint_positions = None
        self.joint_velocities = None

        self.left_shoulder_camera_extrinsics = None
        self.right_shoulder_camera_extrinsics = None
        self.wrist_camera_extrinsics = None
        self.front_camera_extrinsics = None
        self.overhead_camera_extrinsics = None

        self.left_shoulder_camera_intrinsics = None
        self.right_shoulder_camera_intrinsics = None
        self.wrist_camera_intrinsics = None
        self.front_camera_intrinsics = None
        self.overhead_camera_intrinsics = None

        self.left_shoulder_camera_near_far = None
        self.right_shoulder_camera_near_far = None
        self.wrist_camera_near_far = None
        self.front_camera_near_far = None
        self.overhead_camera_near_far = None

        self.task_low_dim_state = None  # Jian 这个不知道是什么


class JianRLBenchDataConverter:
    '''
    convert the RLBench data of RVT-2 to hdf5 file
    why?
    make it easier to load the data!
    '''
    # TODO: wrist camera's extrinsics are not correctly loaded

    def __init__(self, data_dir=None, save_dir=None):

        #

        self.data_dir = data_dir
        self.save_dir = save_dir
        self.folders = sorted([d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))], key=natural_sort_key)
        self.idx = None
        pass

    def save_one_episode(self, idx):
        self.idx = idx
        current_folder = self.folders[idx]
        self.current_folder_path = os.path.join(self.data_dir, current_folder)
        self.data_container = JianRLBenchDataStructure()

        self._load_images()
        self._load_low_dim_obs()
        self._load_variation_descriptions()
        self._load_variation_numer()

        self._save_as_hdf5()

    def _load_images(self):

        def load_images_from_folder(folder, dpeth_flag=False):
            images = []
            for filename in sorted(os.listdir(folder), key=natural_sort_key):
                img_path = os.path.join(folder, filename)
                if os.path.isfile(img_path) and img_path.endswith(('.png', '.jpg', '.jpeg')):
                    img = Image.open(img_path)
                    if dpeth_flag:
                        this_img = image_to_float_array(np.array(img, dtype=np.uint8))
                        pass
                    else:
                        this_img = np.array(img, dtype=np.uint8).transpose(2, 0, 1)  # Convert to CxHxW
                    images.append(this_img)

            return np.stack(images, axis=0)

        self.data_container.left_shoulder_rgb = load_images_from_folder(os.path.join(self.current_folder_path, 'left_shoulder_rgb'))
        self.data_container.left_shoulder_depth = load_images_from_folder(os.path.join(self.current_folder_path, 'left_shoulder_depth'), dpeth_flag=True)
        self.data_container.left_shoulder_mask = load_images_from_folder(os.path.join(self.current_folder_path, 'left_shoulder_mask'))

        self.data_container.right_shoulder_rgb = load_images_from_folder(os.path.join(self.current_folder_path, 'right_shoulder_rgb'))
        self.data_container.right_shoulder_depth = load_images_from_folder(os.path.join(self.current_folder_path, 'right_shoulder_depth'), dpeth_flag=True)
        self.data_container.right_shoulder_mask = load_images_from_folder(os.path.join(self.current_folder_path, 'right_shoulder_mask'))

        self.data_container.wrist_rgb = load_images_from_folder(os.path.join(self.current_folder_path, 'wrist_rgb'))
        self.data_container.wrist_depth = load_images_from_folder(os.path.join(self.current_folder_path, 'wrist_depth'), dpeth_flag=True)
        self.data_container.wrist_mask = load_images_from_folder(os.path.join(self.current_folder_path, 'wrist_mask'))

        self.data_container.front_rgb = load_images_from_folder(os.path.join(self.current_folder_path, 'front_rgb'))
        self.data_container.front_depth = load_images_from_folder(os.path.join(self.current_folder_path, 'front_depth'), dpeth_flag=True)
        self.data_container.front_mask = load_images_from_folder(os.path.join(self.current_folder_path, 'front_mask'))

        self.data_container.overhead_rgb = load_images_from_folder(os.path.join(self.current_folder_path, 'overhead_rgb'))
        self.data_container.overhead_depth = load_images_from_folder(os.path.join(self.current_folder_path, 'overhead_depth'), dpeth_flag=True)
        self.data_container.overhead_mask = load_images_from_folder(os.path.join(self.current_folder_path, 'overhead_mask'))

    def _load_low_dim_obs(self):
        rlbench_low_dim_obs = self._load_pkl_file(os.path.join(self.current_folder_path, 'low_dim_obs.pkl'))
        gripper_joint_positions = []
        gripper_matrix = []
        gripper_open = []
        gripper_touch_forces = []

        joint_forces = []
        joint_positions = []
        joint_velocities = []

        left_shoulder_camera_extrinsics = []
        right_shoulder_camera_extrinsics = []
        wrist_camera_extrinsics = []
        front_camera_extrinsics = []
        overhead_camera_extrinsics = []

        left_shoulder_camera_intrinsics = []
        right_shoulder_camera_intrinsics = []
        wrist_camera_intrinsics = []
        front_camera_intrinsics = []
        overhead_camera_intrinsics = []

        left_shoulder_camera_near_far = []
        right_shoulder_camera_near_far = []
        wrist_camera_near_far = []
        front_camera_near_far = []
        overhead_camera_near_far = []

        task_low_dim_state = []  # Jian 这个不知道是什么

        for this_observation in rlbench_low_dim_obs._observations:
            gripper_joint_positions.append(this_observation.gripper_joint_positions)
            gripper_matrix.append(this_observation.gripper_matrix)
            gripper_open.append(this_observation.gripper_open)
            gripper_touch_forces.append(this_observation.gripper_touch_forces)

            joint_forces.append(this_observation.joint_forces)
            joint_positions.append(this_observation.joint_positions)
            joint_velocities.append(this_observation.joint_velocities)

        this_observation = rlbench_low_dim_obs._observations[0]
        left_shoulder_camera_extrinsics.append(this_observation.misc['left_shoulder_camera_extrinsics'])
        right_shoulder_camera_extrinsics.append(this_observation.misc['right_shoulder_camera_extrinsics'])
        wrist_camera_extrinsics.append(this_observation.misc['wrist_camera_extrinsics'])
        front_camera_extrinsics.append(this_observation.misc['front_camera_extrinsics'])
        overhead_camera_extrinsics.append(this_observation.misc['overhead_camera_extrinsics'])

        left_shoulder_camera_intrinsics.append(this_observation.misc['left_shoulder_camera_intrinsics'])
        right_shoulder_camera_intrinsics.append(this_observation.misc['right_shoulder_camera_intrinsics'])
        wrist_camera_intrinsics.append(this_observation.misc['wrist_camera_intrinsics'])
        front_camera_intrinsics.append(this_observation.misc['front_camera_intrinsics'])
        overhead_camera_intrinsics.append(this_observation.misc['overhead_camera_intrinsics'])

        left_shoulder_camera_near_far.append(np.array([this_observation.misc['left_shoulder_camera_near'], this_observation.misc['left_shoulder_camera_far']]))
        right_shoulder_camera_near_far.append(np.array([this_observation.misc['right_shoulder_camera_near'], this_observation.misc['right_shoulder_camera_far']]))
        wrist_camera_near_far.append(np.array([this_observation.misc['wrist_camera_near'], this_observation.misc['wrist_camera_far']]))
        front_camera_near_far.append(np.array([this_observation.misc['front_camera_near'], this_observation.misc['front_camera_far']]))
        overhead_camera_near_far.append(np.array([this_observation.misc['overhead_camera_near'], this_observation.misc['overhead_camera_far']]))

        task_low_dim_state.append(this_observation.task_low_dim_state)
        pass

        self.data_container.gripper_joint_positions = np.array(gripper_joint_positions, dtype=np.float32)
        self.data_container.gripper_matrix = np.array(gripper_matrix, dtype=np.float32)
        self.data_container.gripper_open = np.array(gripper_open, dtype=np.float32)
        self.data_container.gripper_touch_forces = np.array(gripper_touch_forces, dtype=np.float32)

        self.data_container.joint_forces = np.array(joint_forces, dtype=np.float32)
        self.data_container.joint_positions = np.array(joint_positions, dtype=np.float32)
        self.data_container.joint_velocities = np.array(joint_velocities, dtype=np.float32)

        self.data_container.left_shoulder_camera_extrinsics = np.array(left_shoulder_camera_extrinsics, dtype=np.float32)
        self.data_container.right_shoulder_camera_extrinsics = np.array(right_shoulder_camera_extrinsics, dtype=np.float32)
        self.data_container.wrist_camera_extrinsics = np.array(wrist_camera_extrinsics, dtype=np.float32)
        self.data_container.front_camera_extrinsics = np.array(front_camera_extrinsics, dtype=np.float32)
        self.data_container.overhead_camera_extrinsics = np.array(overhead_camera_extrinsics, dtype=np.float32)

        self.data_container.left_shoulder_camera_intrinsics = np.array(left_shoulder_camera_intrinsics, dtype=np.float32)
        self.data_container.right_shoulder_camera_intrinsics = np.array(right_shoulder_camera_intrinsics, dtype=np.float32)
        self.data_container.wrist_camera_intrinsics = np.array(wrist_camera_intrinsics, dtype=np.float32)
        self.data_container.front_camera_intrinsics = np.array(front_camera_intrinsics, dtype=np.float32)
        self.data_container.overhead_camera_intrinsics = np.array(overhead_camera_intrinsics, dtype=np.float32)

        self.data_container.left_shoulder_camera_near_far = np.array(left_shoulder_camera_near_far, dtype=np.float32)
        self.data_container.right_shoulder_camera_near_far = np.array(right_shoulder_camera_near_far, dtype=np.float32)
        self.data_container.wrist_camera_near_far = np.array(wrist_camera_near_far, dtype=np.float32)
        self.data_container.front_camera_near_far = np.array(front_camera_near_far, dtype=np.float32)
        self.data_container.overhead_camera_near_far = np.array(overhead_camera_near_far, dtype=np.float32)

        self.data_container.task_low_dim_state = np.array(task_low_dim_state, dtype=np.float32)

        pass

    def _load_variation_descriptions(self):
        rlbench_variation_descriptions = self._load_pkl_file(os.path.join(self.current_folder_path, 'variation_descriptions.pkl'))
        self.data_container.variation_descriptions = np.array(rlbench_variation_descriptions)

    def _load_variation_numer(self):
        rlbench_variation_number = self._load_pkl_file(os.path.join(self.current_folder_path, 'variation_number.pkl'))
        self.data_container.variation_number = np.array(rlbench_variation_number)
        pass

    def _load_pkl_file(self, file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data

    def _save_as_hdf5(self):
        if self.save_dir is None:
            KeyError("Please specify the save directory")
        number = self.idx
        save_path = os.path.join(self.save_dir, f"episode_{number}.h5")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with h5py.File(save_path, 'w') as f:
            for attribute_name in self.data_container.__dict__:
                try:
                    f.create_dataset(attribute_name, data=self.data_container.__dict__[attribute_name], dtype=self.data_container.__dict__[attribute_name].dtype)
                except:
                    variable_length_strings = self.data_container.__dict__[attribute_name]
                    dt = h5py.string_dtype(encoding='utf-8')  # Define the variable-length string type
                    dset = f.create_dataset('variation_descriptions', (len(variable_length_strings),), dtype=dt)
                    dset[:] = variable_length_strings
                    pass


if __name__ == "__main__":
    # TODO: parser
    pass
    # multiprocessing
    # folder_path = "/media/jian/data/rlbench_original/val/"
    # tail = "all_variations/episodes"
    # folders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]

    # # 因为，class是一次性load全部，所以不方便mp，先每次create一个试一下

    # def save_one_episode(idx):
    #     folder = folders[idx]
    #     data_dir = os.path.join(folder_path, folder, folder, tail)
    #     save_dir = os.path.join("/media/jian/data/rlbench_hdf5/val/", folder)

    #     episode_number = len(os.listdir(data_dir))
    #     convertor = JianRLBenchDataConverter(data_dir=data_dir, save_dir=save_dir)

    #     for i in range(episode_number):
    #         convertor.save_one_episode(i)

    # with multiprocessing.Pool(processes=18) as pool, tqdm(total=18) as pbar:
    #     for result in pool.imap_unordered(save_one_episode, range(0, 18)):
    #         pbar.update()
    # /multiprocessing
