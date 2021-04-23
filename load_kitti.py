import numpy as np
from tqdm import tqdm
import os

def load(kitti_seq=0):
    assert kitti_seq in range(0, 11)
    print('Loading optical flow and ground truth from KITTI%02d' % kitti_seq)
    print('Note: skipping gt at time 0 since we have no features yet')
    seq_folder = '/media/pjaworsk/5686139C86137C25/tb_storage/datasets/KITTI/data_odometry_color/dataset/sequences'
    # kitti_seq = 3

    vo_dir = '%s/%02d/image_2/flow' %(seq_folder, kitti_seq)
    files = os.listdir(vo_dir)
    # print('all: ', files[:5])
    files[:] = [name for name in files if any(sub in name for sub in ['.npy'])]
    # print('imgs only: ', files[:5])
    files = sorted(files)
    # print('sorted!: ', files[:5])
    flow_array = []
    mxs = []
    for ii in tqdm(range(0, len(files))):
        flow = np.load('%s/%s' % (vo_dir, files[ii]))
        # print('flow: ', flow.shape)
        shape = flow.shape
        flow = flow.reshape(shape[0], shape[1] * shape[2])
        # print('flow: ', flow.shape)
        shape = flow.shape
        flow = flow.reshape(shape[0] * shape[1])
        # print('flow: ', flow.shape)
        flow_array.append(flow)
        mxs.append(max(flow))
    flow_array = np.asarray(flow_array)

    pose_folder = '%s/../poses' % seq_folder
    vel_gt = np.load('%s/%02d_vel.npz' % (pose_folder, kitti_seq))['vel'][1:]

    # print('gt: ', vel_gt.shape)
    # print('flow: ', flow_array.shape)
    assert (vel_gt.shape[0] == flow_array.shape[0])

    return flow_array, vel_gt

if __name__ == '__main__':
    load(0)
