
from itertools import chain, accumulate
from PIL import Image
import os
import re
import numpy as np
import pickle

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


class DataConvertor():
    def __init__(self):
        ''' just to make code more organized '''
        pass

    def load_pickle(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            return data

    def locate_idx_of_task_episode_frame(self, dataset_shape_list: list, idx: int):
        '''
        idx: the global index of the frame
        '''

        def method1(dataset_shape_list, idx):
            '''
            Dont use it, not correct
            '''
            flattened = list(accumulate(chain.from_iterable(dataset_shape_list)))

            # Reorganize into the original shape
            accumulated_shape = []
            index = 0

            for sublist in dataset_shape_list:
                length = len(sublist)
                accumulated_shape.append(flattened[index:index + length])
                index += length

            for list_frames in accumulated_shape:
                if idx < list_frames[-1]:
                    task_idx = accumulated_shape.index(list_frames)
                    print('idx=', idx, 'list[-1]=', list_frames[-1])
                    for frames_num in list_frames:
                        if idx < frames_num:
                            episode_idx = list_frames.index(frames_num) - 1
                            print('idx=', idx, 'frames_num=', frames_num)
                            break
                    frame_idx = idx - list_frames[episode_idx] if episode_idx != 0 else idx
                    print(task_idx, episode_idx, frame_idx)
                    break
            return task_idx, episode_idx, frame_idx

        def method2(dataset_shape_list, idx):
            for i, this_list in enumerate(dataset_shape_list):
                # print('this_list=', this_list)
                for j, frames_num in enumerate(this_list):
                    idx = idx - frames_num
                    if idx < 0:
                        task_idx = i
                        episode_idx = j
                        frame_idx = frames_num + idx
                        # print(task_idx, episode_idx, frame_idx)
                        return task_idx, episode_idx, frame_idx
        return method2(dataset_shape_list, idx)

    def _load_low_dim_obs(self, path):
        rlbench_low_dim_obs = self.load_pickle(path)
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

        # to save
        dict_to_save = dict()
        dict_to_save['gripper_joint_positions'] = gripper_joint_positions
        dict_to_save['gripper_matrix'] = gripper_matrix
        dict_to_save['gripper_open'] = gripper_open
        dict_to_save['gripper_touch_forces'] = gripper_touch_forces

        dict_to_save['joint_forces'] = joint_forces
        dict_to_save['joint_positions'] = joint_positions
        dict_to_save['joint_velocities'] = joint_velocities

        dict_to_save['left_shoulder_camera_extrinsics'] = left_shoulder_camera_extrinsics
        dict_to_save['right_shoulder_camera_extrinsics'] = right_shoulder_camera_extrinsics
        dict_to_save['wrist_camera_extrinsics'] = wrist_camera_extrinsics
        dict_to_save['front_camera_extrinsics'] = front_camera_extrinsics
        dict_to_save['overhead_camera_extrinsics'] = overhead_camera_extrinsics

        dict_to_save['left_shoulder_camera_intrinsics'] = left_shoulder_camera_intrinsics
        dict_to_save['right_shoulder_camera_intrinsics'] = right_shoulder_camera_intrinsics
        dict_to_save['wrist_camera_intrinsics'] = wrist_camera_intrinsics
        dict_to_save['front_camera_intrinsics'] = front_camera_intrinsics
        dict_to_save['overhead_camera_intrinsics'] = overhead_camera_intrinsics

        dict_to_save['left_shoulder_camera_near_far'] = left_shoulder_camera_near_far
        dict_to_save['right_shoulder_camera_near_far'] = right_shoulder_camera_near_far
        dict_to_save['wrist_camera_near_far'] = wrist_camera_near_far
        dict_to_save['front_camera_near_far'] = front_camera_near_far
        dict_to_save['overhead_camera_near_far'] = overhead_camera_near_far

        dict_to_save['task_low_dim_state'] = task_low_dim_state

        return dict_to_save

    def _load_variation_descriptions(self, path, frame_idx):
        dict_return = dict()
        rlbench_variation_descriptions = self.load_pickle(path)
        dict_return['variation_descriptions'] = np.array(rlbench_variation_descriptions)
        return dict_return

    def _load_variation_numer(self, path, frame_idx):
        dict_return = dict()
        rlbench_variation_number = self.load_pickle(path)
        dict_return['variation_number'] = np.array(rlbench_variation_number)
        return dict_return

    def save_one_frame(self, args):

        task_name_all, dataset_shape_list, idx = args
        task_idx, episode_idx, frame_idx = self.locate_idx_of_task_episode_frame(dataset_shape_list, idx)
        task_name = task_name_all[task_idx]
        episode_name = f'episode{episode_idx}'
        frame_png_name = f'{frame_idx}.png'
        episode_dir = f'/media/jian/data/rlbench_original/train/{task_name}/all_variations/episodes/{episode_name}/'
        save_path = f'/media/jian/data/rlbench_frames_0/train/'
        frame_to_save = dict()
        # if os.path.exists(os.path.join(save_path, f'{idx}.pkl')):
        #     return

        # if idx < 89900:
        #     return
        # load all views
        views = [d for d in os.listdir(episode_dir) if os.path.isdir(os.path.join(episode_dir, d))]
        for view in views:
            frame_to_save[view] = np.array(Image.open(os.path.join(episode_dir, view, frame_png_name)))

        # load low_dim_obs
        low_dim_obs_dict = self._load_low_dim_obs(os.path.join(episode_dir, 'low_dim_obs.pkl'))

        # load variation_descriptions
        variation_des_dict = self._load_variation_descriptions(os.path.join(episode_dir, 'variation_descriptions.pkl'), frame_idx)

        # load variation_number
        variation_number_dict = self._load_variation_numer(os.path.join(episode_dir, 'variation_number.pkl'), frame_idx)
        # merge dict
        frame_to_save.update(low_dim_obs_dict)
        frame_to_save.update(variation_des_dict)
        frame_to_save.update(variation_number_dict)

        # save to pickle
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, f'{idx}.pkl'), 'wb') as f:
            pickle.dump(frame_to_save, f)


if __name__ == "__main__":
    folder_path = "/media/jian/data/rlbench_original/train/"
    tail = "all_variations/episodes"
    tasks_name_all = sorted([d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))], key=natural_sort_key)
    num_tasks = len(tasks_name_all)

    dataset_shape_list = []

    for i in range(num_tasks):
        frames_list = []
        task_path = os.path.join(folder_path, tasks_name_all[i], tail)
        episodes_each_task = sorted([d for d in os.listdir(task_path) if os.path.isdir(os.path.join(task_path, d))], key=natural_sort_key)
        num_episodes = len(episodes_each_task)

        for j in range(num_episodes):
            data_dir = os.path.join(folder_path, tasks_name_all[i], tail, episodes_each_task[j], 'front_rgb')
            frames = sorted([d for d in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, d))], key=natural_sort_key)
            frames_num = len(frames)
            frames_list.append(frames_num)
        dataset_shape_list.append(frames_list)

    # # Give an idx, then locate the task episode frame and save to hdf5
    # num_frames = sum([sum(i) for i in dataset_shape_list])
    # with multiprocessing.Pool(processes=30) as pool, tqdm(total=num_frames) as pbar:
    #     args = [(tasks_name_all, dataset_shape_list, i) for i in range(0, num_frames)]
    #     for result in pool.imap_unordered(DataConvertor().save_one_frame, args):
    #         pbar.update()

    pickle.dump(dataset_shape_list, open('/media/jian/data/rlbench_frames_0/dataset_shape_list.pkl', 'wb'))
