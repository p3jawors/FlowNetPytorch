import numpy
import matplotlib.pyplot as plt
import load_kitti

def run(loc, seq, dt=0.1):
    # KITTI fps is 10 so dt=0.1
    vel = np.load(loc)['vel']
    pos = np.cum(vel*dt)
    _, gt = load_kitti(seq)
    print('vel shape: ', vel.shape)
    print('pos shape: ', pos.shape)
    print('gt shape: ', gt.shape)
    gt = gt[seq]
    plt.figure()
    plt.title('KITTI%02d' % seq)
    plt.subplot(221)
    plt.plot(gt[0], 'gt_x')
    pl.plot(pos[0], 'x')
    plt.legend()
    plt.subplot(222)
    plt.plot(gt[1], 'gt_y')
    plt.legend()
    plt.subplot(223)
    plt.plot(gt[2], 'gt_z')
    ax = plt.subplot(224, projection='3d')
    plt.plot(gt[0], gt[1], gt[2], label='gt')
    plt.plot(pos[0], pos[1], pos[2], label='results')
    plt.legend()
    plt.show()
