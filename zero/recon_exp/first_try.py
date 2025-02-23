import open3d as o3d
import numpy as np
import cv2
import pickle


test_data_path = "/media/jian/ssd4t/zero/1_Data/A_Selfgen/train/with_sem/insert_onto_square_peg_peract/variation0/episodes/episode0/data.pkl"


with open(test_data_path, 'rb') as f:
    data = pickle.load(f)
print(data.keys())


num_cam: int
RECON_DATA_TEMPLATE = {
    'rgb': np.zeros((num_cam, 3, 512, 512), dtype=np.uint8),
    'xyz': np.zeros((num_cam, 3, 512, 512), dtype=np.float32),
    'mask': np.zeros((num_cam, 3, 512, 512), dtype=np.uint8),
}


class ReconSystem:
    def __init__(self):
        pass

    def get_first_frame(self, data):
        pass

    def input(self, data):
        pass

    def output(self):
        pass

    def _segment_and_match(self, GTmask=None):
        '''
        ideally, this function should segment the object without gt mask
        '''
        if GTmask is not None:
            mask = GTmask
        else:
            raise NotImplementedError('general segmentation')  # TODO
        return mask


if __name__ == '__main__':
    pass
