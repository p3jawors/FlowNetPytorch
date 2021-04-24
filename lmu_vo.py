import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image, display
import tensorflow as tf
import timeit

import keras_lmu
import load_kitti

KITTI_SEQ = 4

# NOTE loading and formatting dataset
seed = 0
tf.random.set_seed(seed)
np.random.seed(seed)
rng = np.random.RandomState(seed)

# load kitti dataset and a modified gt that gets vel from the pos given by kitti
# input shape of (n_frames/(n_frames_per_theta_window), feature_size*n_frames_per_theta_window, 1)
# NOTE currently not set, so n_frames_per_theta_window=1
try:
    op_flow, gt = load_kitti.load(KITTI_SEQ)
    # np.savez('kitti04', op_flow=op_flow, gt=gt)
except:
    print('Thanks for taking a look at this, make sure you put the npz file in the same directory ;)')
    data = np.load('kitti04.npz')
    op_flow = data['op_flow']
    gt = data['gt']

n_steps = op_flow.shape[0]

# split up 60% train 20% test 20% valid
train_flow = op_flow[:int(n_steps*0.6)]
test_flow = op_flow[int(n_steps*0.6):int(n_steps*0.8)]
valid_flow = op_flow[int(n_steps*0.8):]

train_gt = gt[:int(n_steps*0.6)]
test_gt = gt[int(n_steps*0.6):int(n_steps*0.8)]
valid_gt = gt[int(n_steps*0.8):]


#TODO currently looking one scene at a time, will need a constant normalization
# scene 1 max vo is ~170
max_flow = 170
train_flow = train_flow / max_flow
test_flow = test_flow / max_flow
valid_flow = valid_flow / max_flow

print(
    f"Training inputs shape: {train_flow.shape}, "
    f"Training targets shape: {train_gt.shape}"
)
print(
    f"Validation inputs shape: {valid_flow.shape}, "
    f"Validation targets shape: {valid_gt.shape}"
)
print(f"Testing inputs shape: {test_flow.shape}, Testing targets shape: {test_gt.shape}")

#NOTE defining the model
print('train flow shape: ', train_flow.shape)
shape = train_flow.shape
n_pixels = np.prod(train_flow.shape[1:])
kernel = (1, 2, 2)
# kernel = (2, 2)

lmu_layer = tf.keras.layers.RNN(
    keras_lmu.LMUCell(
        memory_d=1,
        order=256,
        theta=n_pixels/np.prod(kernel), #  this keeps one seq in theta
        hidden_cell=tf.keras.layers.SimpleRNNCell(212),
        hidden_to_memory=False,
        memory_to_memory=False,
        input_to_hidden=True,
        kernel_initializer="ones",
    )
)
# max_pool_layer = tf.keras.layers.MaxPooling3D(
#         pool_size=(1, 2, 2), strides=None, padding='valid')

# conv3_layer = tf.keras.layers.Conv2D(
conv3_layer = tf.keras.layers.Conv3D(
    filters=2, kernel_size=kernel, strides=kernel, padding='valid'
)

# flatten_layer = tf.keras.layers.Flatten(data_format='channels_last')
flatten_layer = tf.keras.layers.Reshape((shape[0], n_pixels))

# TensorFlow layer definition
inputs = tf.keras.Input(shape)
conv3 = conv3_layer(inputs)
flatten = flatten_layer(conv3)
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
do_training = True
batch_size = 100
epochs = 100

saved_weights_fname = "./KITTI%02d-weights.hdf5" % KITTI_SEQ
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=saved_weights_fname, monitor="val_loss", verbose=1, save_best_only=True
    ),
]

start_time = timeit.default_timer()
if do_training:
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

else:
    display(Image(filename="KITTI%02d-training.png" % KITTI_SEQ))

#NOTE Test the model
model.load_weights(saved_weights_fname)
accuracy = model.evaluate(test_flow, test_gt)[1] * 100
print(f"Test accuracy: {round(accuracy, 2):0.2f}%")
