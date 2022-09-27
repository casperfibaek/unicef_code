import numpy as np
import tensorflow as tf
import numpy as np
import gc
from utils import SaveBestModel, OverfitProtection, ConvBlock, ReductionBlock, ExpansionBlock

MODEL_FOLDER = "./models/"
MODEL_NAME = "SuperResSentinel_v20"
TRAIN_DATASET = "./train_superres.npz"
TEST_SIZE = 0.2
BASE_LR = 1e-4
VAL_BS = 256

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
  except RuntimeError as e:
    print(e)

def create_model(
    input_shape_low,
    input_shape_rgb,
    output_channels=1,
    activation="relu",
    activation_output="relu",
    kernel_initializer="glorot_normal",
    name="base",
    size=64,
    depth=1,
):
    """ Get a super-resolution model. """
    SIZE_SMALL = size // 2
    SIZE_MEDIUM = size
    SIZE_LARGE = size + SIZE_SMALL
    POOL_AVG = 7

    model_input_low = tf.keras.Input(shape=(input_shape_low), name=f"{name}_input_low")
    model_input_rgb = tf.keras.Input(shape=(input_shape_rgb), name=f"{name}_input_rgb")

    # The idea here being: Improve the generalisability of the model by transposing the target band to the RGB bands.
    rgb_mean = tf.math.reduce_mean(tf.reduce_mean(model_input_rgb, axis=-1, keepdims=True), name="rgb_mean")
    low_mean = tf.math.reduce_mean(model_input_low, name="low_mean")
    dif_mean = tf.math.divide(rgb_mean, low_mean + tf.keras.backend.epsilon(), name="in_ratio")
    low_adj = tf.math.multiply(model_input_low, dif_mean, name="low_mean_adj")

    conv_rgb = ConvBlock(model_input_rgb, size=SIZE_SMALL, depth=depth, residual=False, activation=activation, kernel_initializer=kernel_initializer)
    conv_rgb = ConvBlock(conv_rgb, size=SIZE_SMALL, depth=depth, residual=True, activation=activation, kernel_initializer=kernel_initializer)
    
    conv_low = tf.keras.layers.Conv2D(size, SIZE_SMALL, padding="same", activation=activation, kernel_initializer=kernel_initializer)(low_adj)
    conv_low = ConvBlock(conv_low, size=SIZE_SMALL, depth=depth, residual=False, activation=activation, kernel_initializer=kernel_initializer)
    conv_low = ConvBlock(conv_low, size=SIZE_SMALL, depth=depth, residual=True, activation=activation, kernel_initializer=kernel_initializer)

    conv = tf.keras.layers.Concatenate()([conv_rgb, conv_low])

    conv_skip1 = ConvBlock(conv, size=SIZE_SMALL, depth=depth, residual=True, activation=activation, kernel_initializer=kernel_initializer)
    redu = ReductionBlock(conv_skip1, activation=activation, kernel_initializer=kernel_initializer)

    conv = ConvBlock(redu, size=SIZE_MEDIUM, depth=depth, residual=False, activation=activation, kernel_initializer=kernel_initializer)
    conv_skip2 = ConvBlock(conv, size=SIZE_MEDIUM, depth=depth, residual=True, activation=activation, kernel_initializer=kernel_initializer)

    redu = ReductionBlock(conv_skip2, activation=activation, kernel_initializer=kernel_initializer)

    conv = ConvBlock(redu, size=SIZE_LARGE, depth=depth + 1, residual=False, activation=activation, kernel_initializer=kernel_initializer)
    expa = ExpansionBlock(conv, activation=activation, kernel_initializer=kernel_initializer)

    merg = tf.keras.layers.Concatenate()([expa, conv_skip2])
    conv = ConvBlock(merg, size=SIZE_MEDIUM, depth=depth, residual=False, activation=activation, kernel_initializer=kernel_initializer)

    expa = ExpansionBlock(conv, activation=activation, kernel_initializer=kernel_initializer)
    merg = tf.keras.layers.Concatenate()([expa, conv_skip1])

    conv = ConvBlock(merg, size=SIZE_SMALL, depth=depth, residual=False, activation=activation, kernel_initializer=kernel_initializer)

    main_output = tf.keras.layers.Conv2D(
        output_channels,
        kernel_size=3,
        padding="same",
        activation=activation_output,
        kernel_initializer=kernel_initializer,
        name="main_output",
        dtype="float32",
    )(conv)

    conf_output = tf.keras.layers.Conv2D(
        output_channels,
        kernel_size=3,
        padding="same",
        activation="sigmoid",
        kernel_initializer=kernel_initializer,
        name="main_output_conf",
        dtype="float32",
    )(conv)

    out_mean_2D = tf.keras.layers.AveragePooling2D(pool_size=(POOL_AVG, POOL_AVG), strides=(1, 1), padding='same')(main_output)
    low_mean_2D = tf.keras.layers.AveragePooling2D(pool_size=(POOL_AVG, POOL_AVG), strides=(1, 1), padding='same')(model_input_low)
    dif_mean_2D = tf.math.divide(low_mean_2D, out_mean_2D + tf.keras.backend.epsilon(), name="out_ratio")
    high_adj = tf.math.multiply(main_output, dif_mean_2D, name="out_mean_adj")

    model = tf.keras.Model(
        inputs=[model_input_low, model_input_rgb],
        outputs=[high_adj, conf_output],
    )

    return model

model_superres = create_model(
    (64, 64, 1),
    (64, 64, 3),
    output_channels=1,
    activation="relu",
    activation_output="relu",
    kernel_initializer="glorot_uniform",
    name="s2super",
    size=64,
    depth=2,
)


def metric_wrapper(metric):
    def func(trues, preds):
        return metric(trues, preds[0])

    return func


def construct_conf_loss(alpha):
    def conf_loss(trues, preds):

        NIR = preds[0]
        conf = preds[1]

        loss = alpha * (1. - conf) + tf.math.divide(tf.math.abs(trues - NIR), 1. - conf)
        return tf.math.reduce_mean(loss)

    return conf_loss

model_superres.compile(optimizer=tf.optimizers.Adam(learning_rate=BASE_LR), loss=construct_conf_loss(0.1), metrics=metric_wrapper(tf.keras.metrics.mean_absolute_error))
# model_superres = tf.keras.models.load_model(MODEL_FOLDER + MODEL_NAME)

loaded = np.load(TRAIN_DATASET)
rgb = loaded["rgb"]
low = loaded["nir_lr"]
label = loaded["nir"]

random_shuffle = np.random.permutation(len(label))
label = label[random_shuffle]
rgb = rgb[random_shuffle]
low = low[random_shuffle]

limit = 50000
label = label[:limit]
rgb = rgb[:limit]
low = low[:limit]

split_frac = int(rgb.shape[0] * 0.1)
x_train_low = low[:-split_frac]
x_train_rgb = rgb[:-split_frac]
y_train = label[:-split_frac]

x_test_low = low[-split_frac:]
x_test_rgb = rgb[-split_frac:]
y_test = label[-split_frac:]

label = None; rgb = None; low = None; loaded = None

def batch_generator_train(batchsize):
    global x_train_low, x_train_rgb, y_train
    patches_len = y_train.shape[0]
    idx = 0

    while True:
        yield [x_train_low[idx:idx + batchsize], x_train_rgb[idx:idx + batchsize]], y_train[idx:idx + batchsize]
        idx = idx + batchsize

        if idx + batchsize > patches_len:
            idx = 0

def batch_generator_test(batchsize):
    global x_test_low, x_test_rgb, y_test
    patches_len = y_test.shape[0]
    idx = 0

    while True:
        yield [x_test_low[idx:idx + batchsize], x_test_rgb[idx:idx + batchsize]], y_test[idx:idx + batchsize]
        idx = idx + batchsize

        if idx + batchsize > patches_len:
            idx = 0

fits_per_epoch = 10
fits = [
    { "epochs": fits_per_epoch, "bs": 16,  "lr": BASE_LR},
    { "epochs": fits_per_epoch, "bs": 32,  "lr": BASE_LR},
    { "epochs": fits_per_epoch, "bs": 64,  "lr": BASE_LR},
    { "epochs": fits_per_epoch, "bs": 96,  "lr": BASE_LR},
    { "epochs": fits_per_epoch, "bs": 128, "lr": BASE_LR},

    { "epochs": fits_per_epoch, "bs": 64,  "lr": BASE_LR * 0.1},
    { "epochs": fits_per_epoch, "bs": 140, "lr": BASE_LR * 0.1},

    { "epochs": fits_per_epoch, "bs": 64,  "lr": BASE_LR * 0.01},
    { "epochs": fits_per_epoch, "bs": 140, "lr": BASE_LR * 0.01},
]

cur_sum = 0
for nr, val in enumerate(fits):
    fits[nr]["ie"] = cur_sum
    cur_sum += fits[nr]["epochs"]

val_loss = model_superres.evaluate(
    batch_generator_test(VAL_BS),
    batch_size=VAL_BS,
    steps=int(y_test.shape[0] / VAL_BS),
)

best_val_loss = val_loss
save_best_model = SaveBestModel(save_best_metric="val_loss", initial_weights=model_superres.get_weights())

out_epoch_path = None
for idx, fit in enumerate(range(len(fits))):
    use_epoch = fits[fit]["epochs"]
    use_bs = fits[fit]["bs"]
    use_lr = fits[fit]["lr"]
    use_ie = fits[fit]["ie"]

    model_superres.optimizer.lr.assign(use_lr)

    model_superres.fit(
        x=batch_generator_train(use_bs),
        validation_data=batch_generator_test(VAL_BS),
        shuffle=True,
        epochs=use_epoch + use_ie,
        initial_epoch=use_ie,
        batch_size=use_bs,
        validation_batch_size=VAL_BS,
        steps_per_epoch=int(y_train.shape[0] / use_bs),
        validation_steps=int(y_test.shape[0] / VAL_BS),
        use_multiprocessing=True,
        workers=0,
        verbose=1,
        callbacks=[
            save_best_model,
            OverfitProtection(
                patience=3,
                difference=0.20, # 20% overfit allowed
                offset_start=3, # disregard overfit for the first epoch
            ),
        ],
    )

    model_superres.set_weights(save_best_model.best_weights)

    best_val_loss = model_superres.evaluate(
        batch_generator_test(VAL_BS),
        batch_size=VAL_BS,
        steps=int(y_test.shape[0] / VAL_BS),
    )

    out_epoch_path = f"{MODEL_FOLDER}S2Super_v1{str(idx)}"
    model_superres.save(out_epoch_path)
