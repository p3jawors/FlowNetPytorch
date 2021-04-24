import numpy as np
import matplotlib.pyplot as plt
# from IPython.display import Image, display
import tensorflow as tf
import timeit

import keras_lmu
import load_kitti

blue = '\033[94m'
endc = '\033[0m'
green = '\033[92m'
red = '\033[91m'

config = tf.compat.v1.ConfigProto(device_count = {'GPU': 0})
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9

def run(
        KITTI_SEQ, saved_weights_fname, memory_d,
        order, theta, do_training, batch_size=32,
        epochs=100, KITTI_SHAPE=None, feature_shape_post_conv=None,
        plot=False):
    # TODO currently empirically hardcoded based off conv layer
    # feautre_shape_post_conv = (46, 153, 2)
    # TRAINVAL_SEQ = [0, 1, 2, 8, 9]
    # TEST_SEQ = [3, 4, 5, 6, 7, 10]


    # NOTE loading and formatting dataset
    seed = 0
    tf.random.set_seed(seed)
    np.random.seed(seed)
    rng = np.random.RandomState(seed)

    # load kitti dataset and a modified gt that gets vel from the pos given by kitti
    # input shape of (n_frames/(n_frames_per_theta_window), feature_size*n_frames_per_theta_window, 1)
    # NOTE currently not set, so n_frames_per_theta_window=1
    # try:
    op_flow, gt = load_kitti.load(KITTI_SEQ)
    print('KITTI%02d' % KITTI_SEQ)
    print('max flow: ', np.amax(op_flow))
    print('opflow shape: ', op_flow.shape)
    print('gt shape: ', gt.shape)
    # if op_flow.shape != KITTI_SHAPE:
    if op_flow.shape[1] != KITTI_SHAPE[0] or op_flow.shape[2] != KITTI_SHAPE[1] or op_flow.shape[3] != KITTI_SHAPE[2]:
        # print('whats happening?')
        # print('op flow full: ', op_flow.shape)
        # print('KITTI full: ', KITTI_SHAPE)
        # print(op_flow.shape[0])
        # print(KITTI_SHAPE[0])
        # print(op_flow.shape[1])
        # print(KITTI_SHAPE[1])
        # print(op_flow.shape[2])
        # print(KITTI_SHAPE[2])
        # print('op flow shape: ', op_flow.shape)
        # print('KITTI shape: ', KITTI_SHAPE)
        op_flow = op_flow[:, 1:, 2:-2, :]
        print('%s************NOTE*************\nreshaping feature: %s' % (blue, endc), op_flow.shape)
        # np.savez('kitti04', op_flow=op_flow, gt=gt)
    # except:
    #     print('Thanks for taking a look at this, make sure you put the npz file in the same directory ;)')
    #     data = np.load('kitti04.npz')
    #     op_flow = data['op_flow']
    #     gt = data['gt']

    #TODO currently looking one scene at a time, will need a constant normalization
    # max flow for all KITTI scenes is 185.17... add some buffer and use 190
    max_flow = 190
    n_steps = op_flow.shape[0]

    if do_training:
        trainval_split = 0.7
        train_flow = op_flow[:int(n_steps*trainval_split)]
        train_flow = train_flow / max_flow
        valid_flow = op_flow[int(n_steps*trainval_split):]
        valid_flow = valid_flow / max_flow

        train_gt = gt[:int(n_steps*trainval_split)]
        valid_gt = gt[int(n_steps*trainval_split):]

        print(
            f"Training inputs shape: {train_flow.shape}, "
            f"Training targets shape: {train_gt.shape}"
        )
        print(
            f"Validation inputs shape: {valid_flow.shape}, "
            f"Validation targets shape: {valid_gt.shape}"
        )

    else:
        test_flow = op_flow
        test_flow = test_flow / max_flow
        test_gt = gt

        print(f"Testing inputs shape: {test_flow.shape}, Testing targets shape: {test_gt.shape}")

    #NOTE defining the model
    # shape = train_flow.shape
    # n_pixels = np.prod(train_flow.shape[1:])
    # shape = train_flow.shape[1:]
    if do_training:
        flow_shape = train_flow.shape[1:]
    else:
        flow_shape = op_flow.shape[1:]
    n_pixels = np.prod(flow_shape)
    # kernel = (1, 2, 2)
    kernel = (2, 2)

    reshape_layer = tf.keras.layers.Reshape(KITTI_SHAPE)

    lmu_layer = tf.keras.layers.RNN(
        keras_lmu.LMUCell(
            memory_d=memory_d,
            order=order,
            # theta=n_pixels/np.prod(kernel), #  this keeps one seq in theta
            # theta=np.prod(KITTI_SHAPE), #  this keeps one seq in theta
            theta=theta,
            hidden_cell=tf.keras.layers.SimpleRNNCell(212),
            hidden_to_memory=False,
            memory_to_memory=False,
            input_to_hidden=True,
            kernel_initializer="ones",
        )
    )
    # max_pool_layer = tf.keras.layers.MaxPooling3D(
    #         pool_size=(1, 2, 2), strides=None, padding='valid')

    conv_layer = tf.keras.layers.Conv2D(
        filters=2, kernel_size=kernel, strides=kernel, padding='valid'
    )

    # TODO shape is currently hardcoded because I don't know how to get conv output shape
    # flatten_layer = tf.keras.layers.Reshape((1, 46*153*2))
    flatten_layer = tf.keras.layers.Reshape((1, np.prod(feature_shape_post_conv)))

    # TensorFlow layer definition
    inputs = tf.keras.Input(flow_shape)
    conv = conv_layer(inputs)
    # reshape = reshape_layer(inputs)
    # conv = conv_layer(reshape)
    flatten = flatten_layer(conv)
    lmus = lmu_layer(flatten)
    outputs = tf.keras.layers.Dense(3)(lmus)

    # TensorFlow model definition
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    for layer in model.layers:
        print(layer.output_shape)
    print('compiling model')
    model.compile(
        # loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        loss='mean_squared_error',
        optimizer="adam",
        metrics=["accuracy"],
    )
    model.summary()

    #NOTE Training
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=saved_weights_fname, monitor="val_loss", verbose=1, save_best_only=True
        ),
    ]

    if do_training:
        start_time = timeit.default_timer()
        print('Starting training...')
        result = model.fit(
            train_flow,
            train_gt,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(valid_flow, valid_gt),
            callbacks=callbacks,
        )

        runtime = timeit.default_timer() - start_time
        print('\n\nTraining ran for %.2f min\n\n' % (runtime/60))

    if do_training:
        plt.figure()
        plt.plot(result.history["val_accuracy"], label="Validation")
        plt.plot(result.history["accuracy"], label="Training")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Post-epoch Training Accuracies")
        plt.xticks(np.arange(epochs), np.arange(1, epochs + 1))
        plt.ylim((0.85, 1.0))  # Restrict range of y axis to (0.85, 1) for readability
        plt.savefig("KITTI%02d-training.png" % KITTI_SEQ)

        val_loss_min = np.argmin(result.history["val_loss"])
        print(
            f"Maximum validation accuracy: "
            f"{round(result.history['val_accuracy'][val_loss_min] * 100, 2):.2f}%"
        )
        return None, val_loss_min

    else:
        # display(Image(filename="KITTI%02d-training.png" % KITTI_SEQ))
        #
        #NOTE Test the model
        model.load_weights(saved_weights_fname)
        accuracy = model.evaluate(test_flow, test_gt)[1] * 100
        out = model.predict(test_flow)
        # print('OUT SHAPE: ', out.shape)
        # accuracy = model.evaluate(test_flow, test_gt)
        # print('ACCURACY: ', accuracy)
        print(f"Test accuracy: {round(accuracy, 2):0.2f}%")
        print('out shape: ', out.shape)
        print('gt shape: ', gt.shape)
        if plot:
            plt.figure()
            plt.subplot(211)
            plt.title('Velocity')
            xs = np.arange(0, len(test_gt))
            plt.scatter(xs, test_gt[:, 0], label='gt x', c='r')
            plt.scatter(xs, test_gt[:, 1], label='gt y', c='g')
            plt.scatter(xs, test_gt[:, 2], label='gt z', c='b')
            plt.plot(out[:, 0], label='pred x', c='r')
            plt.plot(out[:, 1], label='pred y', c='g')
            plt.plot(out[:, 2], label='pred z', c='b')
            plt.legend()
            plt.subplot(212)
            plt.title('Error')
            plt.plot(np.linalg.norm((out-gt), axis=1))
            plt.show()
        return out, accuracy

try:
    from abr_analyze import DataHandler
    dat = DataHandler('syde673-lmu_vo')
except:
    print('%sInstall abr_analyze to save results to hdf4 database%s' % (red, endc))
testnum = 2
batch_size = 32
epochs = 1000
memory_d = 10
order = 256
theta_scale=4 # how many timesteps to keep in dynamical model
show_plot = True
save = False

if not save:
    print('%sWARNING NOT SAVING%s' % (red, endc))
    import time
    time.sleep(2)

# two sizes of images, so shape our features to be same shape between sequences
# we reshape our inputs to this size
KITTI_SHAPE = (93, 307, 2)
# this is the shape of the input after the conv downsampling
feature_shape_post_conv = (46, 153, 2)
# set our theta to a multiple of the downsampled input to choose how many windows to keep in the dynamic model

theta=np.prod(KITTI_SHAPE)*theta_scale, #  this keeps one seq in theta
notes = (
'''
all previous tests had theta_scale of 4\n
testing with theta_scale of 10, epochs 1000, memory 100, order 256
'''
)
# NOTE that changes to trainval or test seq need to be made in run() as well
TRAINVAL_SEQ = [0, 1, 2, 8, 9]
# TRAINVAL_SEQ = [8, 9]
TEST_SEQ = [3, 4, 5, 6, 7, 10]
test_group = 'initial_exploration'

if dat:
    test_params = {
            'testnum': testnum,
            'batch_size': batch_size,
            'epochs': epochs,
            'memory_d': memory_d,
            'order': order,
            'theta': theta,
            'theta_scale': theta_scale,
            'trainval_seq': TRAINVAL_SEQ,
            'test_seq': TEST_SEQ,
            'notes': notes
    }
    dat.save(data=test_params, save_location='%s/%s' % (test_group, testnum), overwrite=True)

# seqs = TRAINVAL_SEQ + TEST_SEQ
seqs = []
# setup to run test inference after every trainval cycle
for seq in TRAINVAL_SEQ:
    seqs.append(seq)
    for x in TEST_SEQ:
        seqs.append(x)


# saved_weights_fname = "./KITTI%02d-weights.hdf5" % KITTI_SEQ
saved_weights_fname = "./test_%04d-epochs_%i-batch_%i-mem_%i-ord_%i-KITTI-weights.hdf5" % (
        testnum, epochs, batch_size, memory_d, order)
# saved_weights_fname = "./test_%04d-epochs_%i-batch_%i-mem_%i-ord_%i-theta_%i-KITTI-weights.hdf5" % (
#         testnum, epochs, batch_size, memory_d, order, theta_scale)
results = []
seqs=[4]
for seq in seqs:
    print('%sTEST PARAMS\n%s%s' % (blue, test_params, endc))
    plot = False
    if seq in TRAINVAL_SEQ:
        do_training = True
        test_type='trainval'
        print('%sRUNNING TRAINING%s' % (red, endc))
    else:
        do_training = False
        test_type = 'test'
        print('%sRUNNING INFERENCE%s' % (green, endc))
        if show_plot:
            plot = True

    output, lossmin_accuracy = run(KITTI_SEQ=seq,
        saved_weights_fname=saved_weights_fname,
        memory_d=memory_d,
        order=order,
        theta=theta,
        do_training=do_training,
        batch_size=batch_size,
        epochs=epochs,
        KITTI_SHAPE=KITTI_SHAPE,
        feature_shape_post_conv=feature_shape_post_conv, #TODO automate this based of cnn params
        plot=plot
    )

    print('lossmin accuracy: ', lossmin_accuracy)
    results.append(lossmin_accuracy)

    if dat and save:
        print('%sData Saved%s' % (green, endc))
        dat.save(
            data={test_type: lossmin_accuracy},
            save_location='%s/%s/KITTI%02d' % (test_group, testnum, seq),
                overwrite=True
        )
        if results is not None:
            dat.save(
                data={'predict': output},
                save_location='%s/%s/KITTI%02d' % (test_group, testnum, seq),
                    overwrite=True
            )

if save:
    dat.save(
        data={'results': results, 'results_seq': seqs},
        save_location='%s/%s/KITTI%02d' % (test_group, testnum),
        overwrite=True
        )

print(results)
