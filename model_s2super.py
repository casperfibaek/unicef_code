import numpy as np
import tensorflow as tf
import numpy as np
from contextlib import redirect_stdout
from utils import SaveBestModel, OverfitProtection, ConvBlock, ReductionBlock, ExpansionBlock

MODEL_FOLDER = "./models/"
MODEL_NAME = "S2Super"
MODEL_VERSION = 1
MODEL_BASE = None

LOG_DIR = "./logs/"
TRAIN_DATASET = "./small_dataset.npz"
TEST_SIZE = 0.2
TEST = True

BASE_LR = 1e-4
VAL_BS = 256
ALPHA = 0.2


# Confidence-loss by Alistair Francis
def construct_conf_loss(alpha):
    def conf_loss(true, pred_and_conf):
        pred, conf = tf.split(pred_and_conf, 2, axis=-1)

        numerator = tf.math.abs(tf.math.subtract(true, pred)) # MAE
        # numerator = tf.math.pow(tf.math.subtract(true, pred), 2.0) # MSE
        # numerator = tf.math.abs(tf.math.divide((tf.math.subtract(true, pred)), true)) # MAPE

        denominator = 1.0 - conf
        loss_pixels = alpha * denominator + tf.math.divide(numerator, denominator)

        return tf.math.reduce_mean(loss_pixels)

    conf_loss.__name__ = "conf_loss"

    return conf_loss


# Must be function, not object metric
def wrap_metric_ignoring_conf(metric, name):
    def metric_func(true, pred_and_conf):
        pred, _conf = tf.split(pred_and_conf, 2, axis=-1)

        return metric(true, pred)

    metric_func.__name__ = name

    return metric_func


def create_model(
    input_shape_low,
    input_shape_rgb,
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

    epsilon = tf.keras.backend.epsilon()

    model_input_low = tf.keras.Input(shape=(input_shape_low), name=f"{name}_input_low")
    model_input_rgb = tf.keras.Input(shape=(input_shape_rgb), name=f"{name}_input_rgb")

    # The idea here being: Improve the generalisability of the model by transposing the target band to the RGB bands.
    mean_rgb = tf.math.reduce_mean(tf.reduce_mean(model_input_rgb, axis=-1, keepdims=True), name="mean_rgb")
    mean_low = tf.math.reduce_mean(model_input_low, name="mean_low")
    mean_dif = tf.math.divide(mean_rgb, tf.math.maximum(mean_low, epsilon), name="mean_dif")
    low_norm = tf.math.multiply(model_input_low, mean_dif, name="low_norm")

    # Pre-merge LOW
    conv_low = tf.keras.layers.Conv2D(size, SIZE_SMALL, padding="same", activation=activation, kernel_initializer=kernel_initializer)(low_norm)
    conv_low = ConvBlock(conv_low, size=SIZE_SMALL, depth=depth, residual=False, activation=activation, kernel_initializer=kernel_initializer)
    conv_low = ConvBlock(conv_low, size=SIZE_SMALL, depth=depth, residual=True, activation=activation, kernel_initializer=kernel_initializer)

    # Pre-merge RGB
    conv_rgb = ConvBlock(model_input_rgb, size=SIZE_SMALL, depth=depth, residual=False, activation=activation, kernel_initializer=kernel_initializer)
    conv_rgb = ConvBlock(conv_rgb, size=SIZE_SMALL, depth=depth, residual=True, activation=activation, kernel_initializer=kernel_initializer)

    conv = tf.keras.layers.Concatenate()([conv_low, conv_rgb])

    # Main block
    conv = ConvBlock(conv, size=SIZE_SMALL, depth=depth, residual=False, activation=activation, kernel_initializer=kernel_initializer)
    conv_skip1 = ConvBlock(conv, size=SIZE_SMALL, depth=depth, residual=True, activation=activation, kernel_initializer=kernel_initializer)
    redu = ReductionBlock(conv_skip1, activation=activation, kernel_initializer=kernel_initializer)

    # 32 x 32
    conv = ConvBlock(redu, size=SIZE_MEDIUM, depth=depth, residual=False, activation=activation, kernel_initializer=kernel_initializer)
    conv_skip2 = ConvBlock(conv, size=SIZE_MEDIUM, depth=depth, residual=True, activation=activation, kernel_initializer=kernel_initializer)
    redu = ReductionBlock(conv_skip2, activation=activation, kernel_initializer=kernel_initializer)

    # 16 x 16
    conv = ConvBlock(redu, size=SIZE_LARGE, depth=depth, residual=False, activation=activation, kernel_initializer=kernel_initializer)
    expa = ExpansionBlock(conv, activation=activation, kernel_initializer=kernel_initializer)

    # 32 x 32
    merg = tf.keras.layers.Concatenate()([expa, conv_skip2])
    conv = ConvBlock(merg, size=SIZE_MEDIUM, depth=depth, residual=False, activation=activation, kernel_initializer=kernel_initializer)
    expa = ExpansionBlock(conv, activation=activation, kernel_initializer=kernel_initializer)
    
    # 64 x64
    merg = tf.keras.layers.Concatenate()([expa, conv_skip1])

    # Main branch
    conv_main = ConvBlock(merg, size=SIZE_SMALL, depth=depth, residual=False, activation=activation, kernel_initializer=kernel_initializer)
    conv_main = ConvBlock(conv_main, size=SIZE_SMALL, depth=depth, residual=True, activation=activation, kernel_initializer=kernel_initializer)

    main_output = tf.keras.layers.Conv2D(
        1,
        kernel_size=3,
        padding="same",
        activation=activation_output,
        kernel_initializer=kernel_initializer,
        dtype="float32",
    )(conv_main)

    mean_pred = tf.math.reduce_mean(main_output, name="mean_pred")
    mean_dif_out = tf.math.divide(mean_low, tf.math.maximum(mean_pred, epsilon), name="mean_dif_out")
    out_norm = tf.math.multiply(main_output, mean_dif_out, name="main_output")

    # Conf Branch
    conv_conf = ConvBlock(merg, size=SIZE_SMALL, depth=depth, residual=False, activation=activation, kernel_initializer=kernel_initializer)
    conv_conf = ConvBlock(conv_conf, size=SIZE_SMALL, depth=depth, residual=True, activation=activation, kernel_initializer=kernel_initializer)

    conf_output = tf.keras.layers.Conv2D(
        1,
        kernel_size=3,
        padding="same",
        activation="sigmoid",
        kernel_initializer=kernel_initializer,
        dtype="float32",
    )(conv_conf)

    conf_output = tf.clip_by_value(conf_output, clip_value_min=epsilon, clip_value_max=1.0 - epsilon, name="main_output_conf")

    # Merge outputs
    concatenated_outputs = tf.keras.layers.Concatenate(axis=-1)([out_norm, conf_output])

    model = tf.keras.Model(
        inputs=[model_input_low, model_input_rgb],
        outputs=concatenated_outputs
    )

    return model

if MODEL_BASE is not None:
    model_superres = tf.keras.models.load_model(MODEL_FOLDER + MODEL_BASE)
else:
    model_superres = create_model(
        (64, 64, 1),
        (64, 64, 3),
        activation="relu",
        activation_output="relu",
        kernel_initializer="glorot_uniform",
        name=f"{MODEL_NAME}_v{MODEL_VERSION}",
        size=64,
        depth=2,
    )

model_superres.compile(
    optimizer=tf.optimizers.Adam(learning_rate=BASE_LR),
    loss=construct_conf_loss(ALPHA),
    metrics=[
        wrap_metric_ignoring_conf(tf.keras.metrics.mean_absolute_error, 'MAE'),
        wrap_metric_ignoring_conf(tf.keras.metrics.mean_squared_error, 'MSE'),
    ],
)

# Save model information
tf.keras.utils.plot_model(
    model_superres,
    to_file=f"./{MODEL_NAME}_{MODEL_VERSION}.png",
    show_shapes=False,
    show_dtype=False,
    show_layer_names=True,
    rankdir='LR',
    expand_nested=False,
    dpi=96,
    layer_range=None,
    show_layer_activations=False
)

with open(f"./{MODEL_NAME}_{MODEL_VERSION}.txt", "w") as f:
    with redirect_stdout(f):
        model_superres.summary()


if TEST:
    loaded = np.load(TRAIN_DATASET)
    x_train_low = loaded['x_train_low']
    x_train_rgb = loaded['x_train_rgb']
    y_train = loaded['y_train']

    x_test_low = loaded['x_test_low']
    x_test_rgb = loaded['x_test_rgb']
    y_test = loaded['x_test'] # Mistake in dataset

else:
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

    split_frac = int(rgb.shape[0] * 0.2)
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

fits = [
    { "epochs": 10, "bs": 16,  "lr": BASE_LR},
    { "epochs":  5, "bs": 32,  "lr": BASE_LR},
    { "epochs":  3, "bs": 64,  "lr": BASE_LR},
    { "epochs":  3, "bs": 96,  "lr": BASE_LR},
    { "epochs":  3, "bs": 128, "lr": BASE_LR},
    { "epochs":  3, "bs": 64,  "lr": BASE_LR * 0.1},
    { "epochs":  3, "bs": 140, "lr": BASE_LR * 0.1},
    { "epochs":  3, "bs": 64,  "lr": BASE_LR * 0.01},
    { "epochs":  3, "bs": 140, "lr": BASE_LR * 0.01},
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
            tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR),
        ],
    )

    model_superres.set_weights(save_best_model.best_weights)

    best_val_loss = model_superres.evaluate(
        batch_generator_test(VAL_BS),
        batch_size=VAL_BS,
        steps=int(y_test.shape[0] / VAL_BS),
    )

    model_superres.save(f"{MODEL_FOLDER}{MODEL_NAME}_v{MODEL_VERSION}_{str(idx)}")

model_superres.save(f"{MODEL_FOLDER}{MODEL_NAME}_v{MODEL_VERSION}")
