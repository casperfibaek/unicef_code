import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage import exposure, transform

from buteo.raster.patches import get_patches, get_kernel_weights, patches_to_array, weighted_median, mad_merge
import buteo

def predict(
    model,
    data_input,
    data_output_proxy,
    number_of_offsets=9,
    tile_size=64,
    borders=True,
    batch_size=None,
    edge_distance=5,
    merge_method="mean",
    verbose=0,
):
    if isinstance(model, str):
        model = tf.keras.models.load_model(model) if isinstance(model, str) else model

    if not isinstance(data_input, list):
        data_input = [data_input]

    assert len(data_input) == len(model.inputs)

    overlaps = []
    for data in data_input:
        overlap, _, _ = get_patches(data, tile_size, number_of_offsets=number_of_offsets, border_check=borders)
        overlaps.append(overlap)

    _, offsets, shapes = get_patches(data_output_proxy, tile_size, number_of_offsets=number_of_offsets, border_check=borders)

    target_shape = list(data_output_proxy.shape)
    if len(target_shape) == 2:
        target_shape.append(len(offsets))
    else:
        target_shape[-1] = len(offsets)
    target_shape.append(1)

    pred_arr = np.zeros(target_shape, dtype="float32")
    conf_arr = np.zeros(target_shape, dtype="float32")

    weights = np.zeros(target_shape, dtype="float32")
    weight_tile = get_kernel_weights(tile_size, edge_distance)

    model = tf.keras.models.load_model(model) if isinstance(model, str) else model

    for idx, offset in enumerate(offsets):
        og_shape = shapes[idx][0:2]; og_shape.append(1)
        og_shape = tuple(og_shape)

        test = []
        for overlap in overlaps:
            test.append(overlap[idx])

        both = model.predict(test, batch_size=batch_size, verbose=verbose)
        predicted,conf = both[...,0][...,np.newaxis], both[...,1][...,np.newaxis]
        pred_reshaped = patches_to_array(predicted, og_shape, tile_size)
        conf_reshaped = patches_to_array(conf, og_shape, tile_size)
        pred_weights = np.tile(weight_tile, (predicted.shape[0], 1, 1))[:, :, :, np.newaxis]
        pred_weights_reshaped = patches_to_array(pred_weights, og_shape, tile_size)

        sx, ex, sy, ey = offset
        pred_arr[sx:ex, sy:ey, idx] = pred_reshaped
        conf_arr[sx:ex, sy:ey, idx] = conf_reshaped
        weights[sx:ex, sy:ey, idx] = pred_weights_reshaped

    weights_sum = np.sum(weights, axis=2)
    weights_norm = (weights[:, :, :, 0] / weights_sum)[:, :, :, np.newaxis]

    if merge_method == "mean":
        return np.average(pred_arr, axis=2, weights=weights_norm), np.average(conf_arr, axis=2, weights=weights_norm)
    elif merge_method == "median":
        return weighted_median(pred_arr, weights_norm), weighted_median(conf_arr, weights_norm)
    elif merge_method == "mad":
        return mad_merge(pred_arr, weights_norm), mad_merge(conf_arr, weights_norm)

    return [pred_arr, conf_arr], weights_norm


if __name__=='__main__':

    SCENEDIR = './scenes'
    MODELPATH = './models/S2Super_v11_MAPE_no-r2-reg'
    model = tf.keras.models.load_model(MODELPATH,compile=False)

    for s in os.listdir(SCENEDIR)[3:]:
        scene = buteo.raster_to_array(os.path.join(SCENEDIR,s))/10000
        rgb = scene[...,[3,2,1]]
        nir_proxy = transform.rescale(transform.rescale(scene[...,4],1/2),2)
        preds,confs = predict(model,[nir_proxy[...,np.newaxis],rgb],nir_proxy[...,np.newaxis],merge_method='mean',number_of_offsets=0,edge_distance=1)

        fig,ax = plt.subplots(2,3,figsize=(15,12))

        ax[0,0].imshow(exposure.equalize_adapthist(np.clip(rgb,0,1),clip_limit=0.02))
        ax[0,1].imshow(nir_proxy,cmap='RdBu',vmin=0,vmax=1)
        ax[0,2].imshow(confs,cmap='RdBu',vmin=confs.min(),vmax=confs.max())


        low_thresh, mid_thresh, high_thresh = np.percentile(confs, [33,66,100])
        print(low_thresh,mid_thresh,high_thresh)
        low_thresh_im = nir_proxy[...,np.newaxis]*0
        mid_thresh_im = nir_proxy[...,np.newaxis]*0
        high_thresh_im = nir_proxy[...,np.newaxis]*0
        low_thresh_im[confs<low_thresh] = preds[confs<low_thresh]
        mid_thresh_im[confs<mid_thresh] = preds[confs<mid_thresh]
        high_thresh_im[confs<high_thresh] = preds[confs<high_thresh]

        ax[1,0].imshow(low_thresh_im,cmap='RdBu',vmin=0,vmax=1)
        ax[1,1].imshow(mid_thresh_im,cmap='RdBu',vmin=0,vmax=1)
        ax[1,2].imshow(high_thresh_im,cmap='RdBu',vmin=0,vmax=1)

        plt.show()