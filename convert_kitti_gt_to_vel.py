import os
import numpy as np
import matplotlib.pyplot as plt

for ii in range(0, 11):
    with open('%02d.txt' % ii) as f:
        # content = f.readlines()
        # content = f.read().split('\n')
        content = []
        while True:
        # for line in f.readline():
            line = f.readline()
            # print(line)
            if not line:
                break
            else:
                content.append([float(x) for x in line.split(' ')])

    # you may also want to remove whitespace characters like `\n` at the end of each line
    # content = [x.strip() for x in content] 
    print(np.asarray(content).shape)
    content = np.array(content).T

    positions = np.array([content[3], content[7], content[11]]).T
    velocities = np.asarray(np.gradient(positions, 0.1, axis=0))
    print('pos: ', positions.shape)
    print(type(velocities))
    print('vel: ', velocities.shape)
    np.savez('%02d_vel' % ii, vel=velocities)
    # plt.figure()
    # plt.subplot(211)
    # plt.plot(positions)
    # plt.subplot(212)
    # plt.plot(velocities)
    # plt.show()
    #
    # plt.figure()
    # # plt.subplot(311)
    # plt.plot(content[3], label='x')
    # plt.plot(content[7], label='y')
    # plt.plot(content[11], label='z')
    # plt.legend()
    # plt.show()
