import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

def load(kitti_seq=0, vel_gt=True):
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
    mns = []
    for ii in tqdm(range(0, len(files))):
        flow = np.load('%s/%s' % (vo_dir, files[ii]))
        # # print('flow: ', flow.shape)
        # shape = flow.shape
        # flow = flow.reshape(shape[0], shape[1] * shape[2])
        # # print('flow: ', flow.shape)
        # shape = flow.shape
        # flow = flow.reshape(shape[0] * shape[1])
        # print('flow: ', flow.shape)
        flow_array.append(flow)
        mxs.append(np.amax(flow))
        mns.append(np.amin(flow))
    flow_array = np.asarray(flow_array)
    print('MAXES: ', max(mxs))
    print('MINS: ', min(mns))

    pose_folder = '%s/../poses' % seq_folder
    if vel_gt:
        gt = np.load('%s/%02d_vel.npz' % (pose_folder, kitti_seq))['vel'][1:]
    else:
        gt = np.load('%s/%02d.npz' % (pose_folder, kitti_seq))['pos'][1:]
    # print('gt: ', vel_gt.shape)
    # print('flow: ', flow_array.shape)
    assert (gt.shape[0] == flow_array.shape[0])

    return flow_array, gt

if __name__ == '__main__':
    for ii in range(0, 11):
        flow, gt = load(ii, False)
        plt.figure()
        plt.subplot(211)
        plt.title('Pos GT')
        plt.legend(['x', 'y', 'z'])
        plt.plot(gt)

        flow, gt = load(ii, True)
        plt.subplot(212)
        plt.title('Vel GT')
        plt.plot(gt)
        plt.legend(['dx', 'dy', 'dz'])
        # plt.show()
        plt.savefig('KITTI%02d-GT.png' % ii)

