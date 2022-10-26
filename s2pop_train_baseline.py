import os
import numpy as np
import buteo as beo
import tensorflow as tf
from glob import glob
from utils import OverfitProtection, struct_mape_metric
from s2pop_predict import predict
from s2pop_model_baseline import create_model

DATA_FOLDER = "D:/CFI/data/sen2pop_v2/"
MODEL_FOLDER = "./models/"

BASE_LR = 1e-4
VAL_BS = 256
INPUT_SHAPE = (64, 64, 10)
ACTIVATION = "swish"
ACTIVATION_OUTPUT = "relu"
KERNEL_INITIALIZER = "glorot_uniform"
NAME = "s2pop_baseline_v4"
SIZE = 64
CARDINALITY = 1
SQUEEZE_RATIO = 8
DENSE_CORE = False
SQUEEZE = False
DEPTH = 1
OPTIMIZER = tf.optimizers.Adam(learning_rate=BASE_LR)
LOSS = "logcosh"
BETA = 0.01
METRICS = ["mse", "mae", struct_mape_metric(BETA), tf.keras.metrics.Precision(BETA), tf.keras.metrics.Recall(BETA)]
PREDICT = True

LIMIT = 50000
VAL_SPLIT = 0.1
TARGET = 0 # Buildings = 0, Roads = 1, 


model_s2pop = create_model(
    INPUT_SHAPE,
    activation=ACTIVATION,
    activation_output=ACTIVATION_OUTPUT,
    kernel_initializer=KERNEL_INITIALIZER,
    name=NAME,
    size=SIZE,
    cardinality=CARDINALITY,
    squeeze_ratio=SQUEEZE_RATIO,
    squeeze=SQUEEZE,
    depth=DEPTH,
    dense_core=DENSE_CORE,
)
model_s2pop.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=METRICS)

save_best_model = tf.keras.models.clone_model(model_s2pop)
save_best_model.build(INPUT_SHAPE)
save_best_model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=METRICS)
save_best_model.set_weights(model_s2pop.get_weights())

x_train = np.load(DATA_FOLDER + "x_train_50000.npz")["data"]
x_test = np.load(DATA_FOLDER + "x_test_5000.npz")["data"]

y_train = np.load(DATA_FOLDER + "y_train_50000.npz")["label"][:, :, :, TARGET][:, :, :, np.newaxis]
y_test = np.load(DATA_FOLDER + "y_test_5000.npz")["label"][:, :, :, TARGET][:, :, :, np.newaxis]

x_train = x_train[:LIMIT, :, :, :]
y_train = y_train[:LIMIT, :, :, :]

val_total = int(x_train.shape[0] * VAL_SPLIT)

x_val = x_train[-val_total:, :, :, :]; x_train = x_train[:-val_total, :, :, :]
y_val = y_train[-val_total:, :, :, :]; y_train = y_train[:-val_total, :, :, :]

fits = [
    # { "epochs": 10, "bs": 64,  "lr": 1e-04},
    { "epochs": 20, "bs": 64,  "lr": 1e-04},
    { "epochs": 10, "bs": 96,  "lr": 1e-04},
    { "epochs": 10, "bs": 128, "lr": 1e-04},
]

cur_sum = 0
for nr, val in enumerate(fits):
    fits[nr]["ie"] = cur_sum
    cur_sum += fits[nr]["epochs"]

best_val_loss = model_s2pop.evaluate(x_val, y_val, verbose=0)[0]

class SaveBestModel(tf.keras.callbacks.Callback):
    def __init__(self):
        pass

    def on_epoch_end(self, _epoch, logs=None):
        global save_best_model, best_val_loss
        val_loss = logs['val_loss']

        if val_loss <= best_val_loss:
            save_best_model.set_weights(self.model.get_weights())

for idx, fit in enumerate(fits):
    use_epoch = fit["epochs"]
    use_bs = fit["bs"]
    use_lr = fit["lr"]
    use_ie = fit["ie"]

    model_s2pop.optimizer.lr.assign(use_lr)

    _history = model_s2pop.fit(
        x=x_train,
        y=y_train,
        validation_data=(x_val, y_val),
        shuffle=True,
        epochs=use_epoch + use_ie,
        initial_epoch=use_ie,
        batch_size=use_bs,
        validation_batch_size=VAL_BS,
        use_multiprocessing=True,
        workers=0,
        verbose=1,
        callbacks=[
            SaveBestModel(),
            OverfitProtection(
                patience=3,
                difference=0.20, # 20% overfit allowed
                offset_start=3, # disregard overfit for the first epoch
            ),
        ],
    )

    model_s2pop.set_weights(save_best_model.get_weights())

    print("Test data:")
    model_s2pop.evaluate(x_test, y_test)

model_s2pop.save(f"D:/CFI/models/{NAME}.model")

if PREDICT:
    images = glob("D:/CFI/predictions/test_images/*.tif")
    for img in images:
        img_name = os.path.splitext(os.path.basename(img))[0]
        
        if len(img_name.split("_")) > 2:
            continue
        
        arr = beo.raster_to_array(img)[:, :, 1:] # loose scl
        predicted = predict(model_s2pop, arr, arr)

        beo.array_to_raster(predicted, reference=img, out_path=f"D:/CFI/predictions/{img_name}_{NAME}_pred.tif")