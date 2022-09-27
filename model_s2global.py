import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision

np.set_printoptions(suppress=True)
mixed_precision.set_global_policy("mixed_float16")

class SaveBestModel(tf.keras.callbacks.Callback):
    def __init__(self, save_best_metric="val_loss", this_max=False, initial_weights=None):
        self.save_best_metric = save_best_metric
        self.max = this_max

        if initial_weights is not None:
            self.best_weights = initial_weights

        if this_max:
            self.best = float("-inf")
        else:
            self.best = float("inf")

    def on_epoch_end(self, _epoch, logs=None):
        metric_value = abs(logs[self.save_best_metric])
        if self.max:
            if metric_value > self.best:
                self.best = metric_value
                self.best_weights = self.model.get_weights()

        else:
            if metric_value < self.best:
                self.best = metric_value
                self.best_weights = self.model.get_weights()


class OverfitProtection(tf.keras.callbacks.Callback):
    def __init__(self, difference=0.1, patience=3, offset_start=3, verbose=True):
        self.difference = difference
        self.patience = patience
        self.offset_start = offset_start
        self.verbose = verbose
        self.count = 0

    def on_epoch_end(self, epoch, logs=None):
        loss = logs['loss']
        val_loss = logs['val_loss']
        
        if epoch < self.offset_start:
            return

        epsilon = 1e-7
        ratio = loss / (val_loss + epsilon)

        if (1.0 - ratio) > self.difference:
            self.count += 1

            if self.verbose:
                print(f"Overfitting.. Patience: {self.count}/{self.patience}")

        elif self.count != 0:
            self.count -= 1
        
        if self.count >= self.patience:
            self.model.stop_training = True

            if self.verbose:
                print(f"Training stopped to prevent overfitting. Difference: {ratio}, Patience: {self.count}/{self.patience}")

def divide_filters(size, width):
    if width > size:
        raise ValueError("Wider than size.")
    
    step = int(size / width)
    missing = size - (width * step)

    batches = [step] * width

    for idx in range(0, missing):
        batches[idx] += 1

    return batches

def ConvBlockBase(input_layer, size, residual=False, activation="relu", kernel_initializer="glorot_normal"):
    if residual:
        size = input_layer.shape[-1]

    conv1 = tf.keras.layers.Conv2D(size, 1, padding="same", activation=activation, kernel_initializer=kernel_initializer)(input_layer)
    conv1 = tf.keras.layers.Conv2D(size, 1, padding="same", activation=activation, kernel_initializer=kernel_initializer, use_bias=False)(conv1)

    conv2 = tf.keras.layers.Conv2D(size, 1, padding="same", activation=activation, kernel_initializer=kernel_initializer)(input_layer)
    conv2 = tf.keras.layers.Conv2D(size, 3, padding="same", activation=activation, kernel_initializer=kernel_initializer, use_bias=False)(conv2)

    conv3 = tf.keras.layers.Conv2D(size, 1, padding="same", activation=activation, kernel_initializer=kernel_initializer)(input_layer)
    conv3 = tf.keras.layers.Conv2D(size, 5, padding="same", activation=activation, kernel_initializer=kernel_initializer, use_bias=False)(conv3)

    merged = tf.keras.layers.Add()([conv1, conv2, conv3])
    merged = tf.keras.layers.BatchNormalization()(merged)
    merged = tf.keras.layers.Activation(activation)(merged)

    if residual:
        return tf.keras.layers.Add()([input_layer, merged])

    return merged

def ConvBlock(input_layer, size, depth=1, width=1, residual=False, activation="relu", kernel_initializer="glorot_normal"):
    
    wide_layers = []

    sizes = divide_filters(size, width)

    if residual:
        wide_layers.append(input_layer)

    for w in range(0, width):
        previous_depth = input_layer

        for d in range(0, depth):
            previous_depth = ConvBlockBase(previous_depth, sizes[w], residual=residual, activation=activation, kernel_initializer=kernel_initializer)

        wide_layers.append(previous_depth)

    if len(wide_layers) > 1:
        if residual:
            return tf.keras.layers.Add()(wide_layers)
        else:
            return tf.keras.layers.Concatenate()(wide_layers)
    
    return wide_layers[0]


def ReductionBlock(
    layer_input,
    activation="relu",
    kernel_initializer="glorot_normal",
):
    """Extract patches from a tf.Layer. Only channel last format allowed."""
    track1 = tf.keras.layers.AveragePooling2D(padding="same")(layer_input)

    track2 = tf.keras.layers.Conv2D(layer_input.shape[-1], kernel_size=1, padding="same", strides=(1, 1), activation=activation, kernel_initializer=kernel_initializer)(layer_input)
    track2 = tf.keras.layers.Conv2D(layer_input.shape[-1], kernel_size=3, padding="same", strides=(2, 2), activation=activation, kernel_initializer=kernel_initializer, use_bias=False)(track2)
    track2 = tf.keras.layers.BatchNormalization()(track2)
    track2 = tf.keras.layers.Activation(activation)(track2)

    return tf.keras.layers.Add()([track1, track2])

def ExpansionBlock(
    layer_input,
    activation="relu",
    kernel_size=3,
    kernel_initializer="glorot_normal",
):
    track1 = tf.keras.layers.Conv2D(layer_input.shape[-1], kernel_size=1, padding="same", strides=(1, 1), activation=activation, kernel_initializer=kernel_initializer)(layer_input)
    track1 = tf.keras.layers.Conv2DTranspose(layer_input.shape[-1], kernel_size=kernel_size, strides=(2, 2), activation=activation, padding="same", kernel_initializer=kernel_initializer, use_bias=False)(track1)
    track1 = tf.keras.layers.BatchNormalization()(track1)
    track1 = tf.keras.layers.Activation(activation)(track1)

    return track1

def get_latlng_predictor(shape, size=64, depth=2, width=1, activation="relu", output_activation="relu", kernel_initializer="glorot_uniform"):
    encoder_inputs = tf.keras.Input(shape=shape)

    size_small = size // 2
    size_medium = size
    size_large = size + size_small
    first_residual = False
    second_residual = True

    conv = ConvBlock(encoder_inputs, size=size_small, depth=depth, width=width, residual=first_residual, activation=activation, kernel_initializer=kernel_initializer)
    conv_outer = ConvBlock(conv, size=size_small, depth=depth, width=width, residual=second_residual, activation=activation, kernel_initializer=kernel_initializer)
    redu = ReductionBlock(conv_outer, activation=activation, kernel_initializer=kernel_initializer)
    conv = ConvBlock(redu, size=size_medium, depth=depth, width=width, residual=first_residual, activation=activation, kernel_initializer=kernel_initializer)
    conv_inner = ConvBlock(conv, size=size_medium, depth=depth, width=width, residual=second_residual, activation=activation, kernel_initializer=kernel_initializer)
    redu = ReductionBlock(conv_inner, activation=activation, kernel_initializer=kernel_initializer)

    conv = ConvBlock(redu, size=size_large, depth=depth, width=width, residual=first_residual, activation=activation, kernel_initializer=kernel_initializer)
    encoded = tf.keras.layers.Conv2D(size_large, 1, padding="same", name="encoder")(conv)
    conv = ConvBlock(encoded, size=size_large, depth=depth, width=width, residual=first_residual, activation=activation, kernel_initializer=kernel_initializer)

    expa = ExpansionBlock(conv, activation=activation, kernel_initializer=kernel_initializer)
    conv = ConvBlock(expa, size=size_medium, depth=depth, width=width, residual=first_residual, activation=activation, kernel_initializer=kernel_initializer)
    conc = tf.keras.layers.Concatenate()([conv, conv_inner])
    conv = ConvBlock(conc, size=size_medium, depth=depth, width=width, residual=first_residual, activation=activation, kernel_initializer=kernel_initializer)
    expa = ExpansionBlock(conv, activation=activation, kernel_initializer=kernel_initializer)
    conv = ConvBlock(expa, size=size_small, depth=depth, width=width, residual=first_residual, activation=activation, kernel_initializer=kernel_initializer)
    conc = tf.keras.layers.Concatenate()([conv, conv_outer])
    conv = ConvBlock(conc, size=size_small, depth=depth, width=width, residual=first_residual, activation=activation, kernel_initializer=kernel_initializer)

    transfer = tf.keras.layers.Conv2D(size_medium, 1, padding="same", name=f"transfer_full_{str(shape[0])}_{str(shape[1])}_{str(size)}")(conv)

    redu = ReductionBlock(transfer, activation=activation, kernel_initializer=kernel_initializer)
    conv = ConvBlock(redu, size=size_medium, depth=depth, width=width, residual=first_residual, activation=activation, kernel_initializer=kernel_initializer)
    conc = tf.keras.layers.Add()([conv, conv_inner])
    redu = ReductionBlock(conc, activation=activation, kernel_initializer=kernel_initializer)
    conv = ConvBlock(redu, size=size_large, depth=depth, width=width, residual=first_residual, activation=activation, kernel_initializer=kernel_initializer)
    compressed = tf.keras.layers.Conv2D(10, 1, padding="same", name=f"transfer_reduced_{str(shape[0] // 2)}_{str(shape[1] // 2)}_10")(conv)

    flat = tf.keras.layers.Flatten()(compressed)
    dens = tf.keras.layers.Dense(512, activation=activation, kernel_initializer=kernel_initializer)(flat)
    dens = tf.keras.layers.Dense(512, activation=activation, kernel_initializer=kernel_initializer)(dens)
    output = tf.keras.layers.Dense(6, activation=output_activation, kernel_initializer=kernel_initializer)(dens)

    return tf.keras.Model(encoder_inputs, output, name="latlng_predictor")

NORMALISE = 10000.0
TILE_SIZE = 64
DEPTH = 4
WIDTH = 1
MODEL_SHAPE = (64, 64, 10)
TEST_SIZE = 0.2
BASE_LR = 0.00001
DATA_FOLDER = "./prep_data/"
MODEL_FOLDER = "./models/"

mirrored_strategy = tf.distribute.MirroredStrategy(
    devices=["/job:localhost/replica:0/task:0/device:GPU:0", "/job:localhost/replica:0/task:0/device:GPU:1"],
    cross_device_ops=tf.distribute.ReductionToOneDevice(),
)

with mirrored_strategy.scope():
# model_latlng = get_latlng_predictor(MODEL_SHAPE, size=TILE_SIZE, depth=DEPTH, width=WIDTH, activation="relu", output_activation="sigmoid", kernel_initializer="glorot_uniform")
    model_latlng_path = f"{MODEL_FOLDER}Sentinel_Global_v5"
    model_latlng = tf.keras.models.load_model(model_latlng_path)
    weights = model_latlng.get_weights()
    model_latlng.compile(optimizer=tf.optimizers.Adam(learning_rate=BASE_LR), loss="mse", metrics=["log_cosh", "mae"])
    model_latlng.set_weights(weights)

    batch_mult = mirrored_strategy.num_replicas_in_sync

    tf_options = tf.data.Options()
    tf_options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

trainable_count = np.sum([tf.keras.backend.count_params(w) for w in model_latlng.trainable_weights])
print(f"Parameters: {str(np.round(trainable_count / 1000000, 3))}m")

training_data_1 = np.load(DATA_FOLDER + "train_global_uint16_p1.npz")
training_data_2 = np.load(DATA_FOLDER + "train_global_uint16_p2.npz")
bands1 = (training_data_1["bands"] / NORMALISE).astype("float32")
bands2 = (training_data_2["bands"] / NORMALISE).astype("float32")
training_bands = np.concatenate([bands1, bands2], axis=0)
bands1 = None
bands2 = None
training_sincos = np.concatenate([training_data_1["sincos"], training_data_2["sincos"]], axis=0)
training_data_1 = None
training_data_2 = None

print("Loaded data.")
print(training_bands.shape)
print(training_sincos.shape)
print(training_bands.dtype)
print(training_sincos.dtype)

test_size = int(training_bands.shape[0] * TEST_SIZE)
x_train = training_bands[:-test_size, :, :, :]
y_train = training_sincos[:-test_size, :]
# train = tf.data.Dataset.from_tensor_slices((x_train, y_train))

x_test = training_bands[-test_size:, :, :, :]
y_test = training_sincos[-test_size:, :]
# test = tf.data.Dataset.from_tensor_slices((x_test, y_test))

def batch_generator(image, label, batchsize):
    random_shuffle = np.random.permutation(len(image))
    patches_len = len(image)
    idx = 0

    while True:
        yield image[random_shuffle][idx:idx + batchsize], label[random_shuffle][idx:idx + batchsize]
        idx = idx + batchsize

        if idx + batchsize > patches_len:
            idx = 0
            random_shuffle = np.random.permutation(len(image))

fits = [
    { "epochs": 5, "bs": 32, "lr": BASE_LR },
]

class_weights = {
    0: 1.2,
    1: 1.2,
    2: 1.0,
    3: 1.0,
    4: 0.5,
    5: 0.5,
} # Weight latitude > longitude, and latlng > month

cur_sum = 0
for nr, val in enumerate(fits):
    fits[nr]["ie"] = cur_sum
    cur_sum += fits[nr]["epochs"]

val_loss = model_latlng.evaluate(
    tf.data.Dataset.from_generator(batch_generator(x_test, y_test, 64)).with_options(tf_options),
    batch_size=64,
    steps=int(y_test.shape[0] / 64),
)

best_val_loss = val_loss
save_best_model = SaveBestModel(save_best_metric="val_loss", initial_weights=model_latlng.get_weights())

for fit in range(len(fits)):
    use_epoch = fits[fit]["epochs"]
    use_bs = fits[fit]["bs"]
    use_lr = fits[fit]["lr"]
    use_ie = fits[fit]["ie"]

    with mirrored_strategy.scope():
        model_latlng.optimizer.lr.assign(use_lr)

        model_latlng.fit(
            x=tf.data.Dataset.from_generator(batch_generator(x_train, y_train, use_bs)).with_options(tf_options),
            validation_data=tf.data.Dataset.from_generator(batch_generator(x_test, y_test, use_bs)).with_options(tf_options),
            class_weight=class_weights,
            steps_per_epoch=int(y_train.shape[0] / use_bs),
            validation_steps=int(y_test.shape[0] / use_bs),
            shuffle=True,
            epochs=use_epoch + use_ie,
            initial_epoch=use_ie,
            batch_size=use_bs,
            use_multiprocessing=True,
            workers=0,
            verbose=1,
            callbacks=[
                save_best_model,
                OverfitProtection(
                    patience=3,
                    difference=0.2, # 20% overfit allowed
                    offset_start=3, # disregard overfit for the first epoch
                ),
            ],

        )

    model_latlng.set_weights(save_best_model.best_weights)

    best_val_loss = model_latlng.evaluate(
        tf.data.Dataset.from_generator(batch_generator(x_test, y_test, 64)).with_options(tf_options),
        batch_size=64,
        steps=int(y_test.shape[0] / 64),
    )
    
    model_latlng.save(f"{MODEL_FOLDER}Sentinel_Global_v6_{str(fit)}")

model_latlng.save(f"{MODEL_FOLDER}Sentinel_Global_v6")
