import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import buteo as beo
from glob import glob
from functools import cmp_to_key


def cmp_items(a, b):
    idx_location = 5
    a_int = int(os.path.splitext(os.path.basename(a))[0].split("_")[idx_location])
    b_int = int(os.path.splitext(os.path.basename(b))[0].split("_")[idx_location])
    if a_int > b_int:
        return 1
    elif a_int == b_int:
        return 0
    else:
        return -1

def ready_image(glob_path, divisions=10, limit=100, smooth=3):
    images = glob(glob_path)
    images.sort(key=cmp_to_key(cmp_items))

    arrays = []
    for img in images[:limit]:
        img = beo.raster_to_array(img)[:, :, 0]
        arrays.append(img)

    ret_arr = None

    arrays_interpolated = []
    for idx, arr in enumerate(arrays):
        arr_from = arrays[idx-1] if idx != 0 else np.zeros_like(arr)

        interpolated = np.linspace(arr_from, arr, divisions, axis=0)
        for i in range(divisions):
            arrays_interpolated.append(interpolated[i, :, :])
    
    if smooth != 0:
        arrays_smoothed = []
        for idx, arr in enumerate(arrays_interpolated):
            to_smooth = []
            if idx < smooth:
                for i in range(smooth):
                    to_smooth.append(arrays_interpolated[idx + i])
                
            elif idx > len(arrays_interpolated) - smooth:
                for i in range(smooth):
                    to_smooth.append(arrays_interpolated[idx - i])
            else:
                for i in range(smooth):
                    if i == 0:
                        to_smooth.append(arrays_interpolated[idx])
                    else:
                        to_smooth.append(arrays_interpolated[idx + i])
                        to_smooth.append(arrays_interpolated[idx - i])

            smoothed = np.mean(to_smooth, axis=0)
            arrays_smoothed.append(smoothed)

        ret_arr = np.array(arrays_smoothed)
    
    else:
        ret_arr = np.array(arrays_interpolated)

    return ret_arr


DIVISIONS = 10
DPI = 130
HEIGHT = 1920
WIDTH = 1080
LIMIT = 136
PRERUN = 3
TEXT_COLOR = "#BBBBBB"
IMG_RAMP = "magma"
IMG_BACKGROUND = "#0f0f0f"
SQR_1 = "#FFC300"
SQR_2 = "#C70039"
FOLDER = "/home/casper/Desktop/UNICEF/"

places = [
    { "id": "0", "name": "San Diego", "name_abr": "san-diego", "blue": [270, 1950], "red": [500, 1000] },
    { "id": "10", "name": "Palm Springs", "name_abr": "palm-springs", "blue": [850, 1250], "red": [1900, 1560] }, 
    { "id": "20", "name": "Lake Isabella", "name_abr": "lake-isabella", "blue": [1820, 1300], "red": [930, 1400] },
    { "id": "30", "name": "Nevada Border", "name_abr": "nevada-border", "blue": [1800, 1750], "red": [1000, 1150] },
    { "id": "40", "name": "Dixon", "name_abr": "dixon", "blue": [1000, 1000], "red": [50, 1400] },
    { "id": "50", "name": "Lake Port", "name_abr": "lake-port", "blue": [1770, 1950], "red": [1000, 1190] },
    { "id": "60", "name": "Eugene", "name_abr": "eugene", "blue": [1600, 1200], "red": [880, 540] },
    { "id": "70", "name": "Covelo", "name_abr": "covelo", "blue": [1000, 830], "red": [700, 50] },
    { "id": "80", "name": "Homestead", "name_abr": "homestead", "blue": [700, 880], "red": [1900, 50] },
    { "id": "90", "name": "Melbourne", "name_abr": "melbourne", "blue": [825, 1475], "red": [1400, 150] },
]

# df = pd.read_csv("/home/casper/Desktop/UNICEF/s2pop_gan_v21_history.csv")
# smape = df["val_s_mape"].values

for idx, place in enumerate(places):
    fig_id = place["id"]
    fig_name = place["name"]
    fig_name_abr = place["name_abr"]

    fig = plt.figure(figsize=(HEIGHT / DPI, WIDTH / DPI), dpi=DPI, facecolor=IMG_BACKGROUND)

    spec = gridspec.GridSpec(nrows=2, ncols=3)

    ax_main = fig.add_subplot(spec[0:2, 0:2])
    ax_sup1 = fig.add_subplot(spec[0, 2])
    ax_sup2 = fig.add_subplot(spec[1, 2])

    ax_main.set_axis_off(); ax_main.set_xticks([]); ax_main.set_yticks([])
    ax_sup1.set_axis_off(); ax_sup1.set_xticks([]); ax_sup1.set_yticks([])
    ax_sup2.set_axis_off(); ax_sup2.set_xticks([]); ax_sup2.set_yticks([])

    ax_main.set_facecolor(IMG_BACKGROUND)
    ax_sup1.set_facecolor(IMG_BACKGROUND)
    ax_sup2.set_facecolor(IMG_BACKGROUND)

    size = 500
    sup1_rect = place["red"]
    sup2_rect = place["blue"]

    lw = 3
    rect1 = patches.Rectangle((sup1_rect[0], sup1_rect[1]), size, size, linewidth=lw, edgecolor=SQR_1, facecolor='none')
    rect2 = patches.Rectangle((sup2_rect[0], sup2_rect[1]), size, size, linewidth=lw, edgecolor=SQR_2, facecolor='none')

    ax_sup1_border = patches.Rectangle((lw / 2, lw / 2), size - (lw + (lw / 2)), size - (lw + (lw / 2)), linewidth=lw, edgecolor=SQR_1, facecolor='none')
    ax_sup2_border = patches.Rectangle((lw / 2, lw / 2), size - (lw + (lw / 2)), size - (lw + (lw / 2)), linewidth=lw, edgecolor=SQR_2, facecolor='none')

    rgb_image = f"{FOLDER}gifs/test_images/north-america_{fig_id}.tif"
    rgb_image = beo.raster_to_array(rgb_image)
    rgb_image = rgb_image[:, :, 1:4] / 10000.0

    q02 = np.quantile(rgb_image, 0.02)
    q99 = np.quantile(rgb_image, 0.99)

    rgb_image = np.clip(rgb_image[:, :, ::-1], q02, q99)
    rgb_image = (rgb_image - q02) / (q99 - q02)

    predictions = ready_image(f"{FOLDER}gifs/s2pop_gan_v21_north-america_{fig_id}_*.tif", divisions=DIVISIONS, limit=LIMIT)

    print(f"Read: {fig_name}")

    red_square_rgb = rgb_image[sup1_rect[1]:sup1_rect[1]+size, sup1_rect[0]:sup1_rect[0]+size, :]
    red_square_pred = predictions[0, sup1_rect[1]:sup1_rect[1] + size, sup1_rect[0]:sup1_rect[0] + size]

    blue_square_rgb = rgb_image[sup2_rect[1]:sup2_rect[1]+size, sup2_rect[0]:sup2_rect[0]+size, :]
    blue_square_pred = predictions[0, sup2_rect[1]:sup2_rect[1] + size, sup2_rect[0]:sup2_rect[0] + size]

    # main
    main_rgb = ax_main.imshow(rgb_image, interpolation="antialiased")
    ax_main.add_patch(rect1)
    ax_main.add_patch(rect2)
    main_pred = ax_main.imshow(predictions[0], vmin=0.0, vmax=1.0, interpolation="antialiased", cmap=IMG_RAMP, alpha=np.zeros_like(predictions[0]))

    # red
    red_rgb = ax_sup1.imshow(red_square_rgb, interpolation="antialiased")
    ax_sup1.add_patch(ax_sup1_border)
    red_pred = ax_sup1.imshow(red_square_pred, vmin=0.0, vmax=1.0, interpolation="antialiased", cmap=IMG_RAMP, alpha=np.zeros_like(red_square_pred))

    # blue
    blue_rgb = ax_sup2.imshow(blue_square_rgb, interpolation="antialiased")
    ax_sup2.add_patch(ax_sup2_border)
    blue_pred = ax_sup2.imshow(blue_square_pred, vmin=0.0, vmax=1.0, interpolation="antialiased", cmap=IMG_RAMP, alpha=np.zeros_like(blue_square_pred))

    times = [0.0] * (DIVISIONS * PRERUN) + list(np.arange(0, LIMIT, 1 / DIVISIONS))

    time_text = ax_main.text(0.05, 0.05, str(round(times[0], 1)), fontsize=15, color=TEXT_COLOR, transform=ax_main.transAxes)
    place_text = ax_main.text(0.5, 0.95, fig_name, fontsize=15, color=TEXT_COLOR, transform=ax_main.transAxes)

    plt.tight_layout()
    fig.subplots_adjust(wspace=-0.345, hspace=0.01, left=0.0, right=1.0, top=1.0, bottom=0.0)
    plt.rcParams["figure.facecolor"] = IMG_BACKGROUND

    def updatefig(j):
        global ax_main, ax_sup1, ax_sup2, rgb_image, predictions, red_rgb, blue_rgb, sup1_rect, sup1_rect, red_pred, blue_pred, time_text, place_text

        nr_preds = predictions.shape[0]

        if j > DIVISIONS * PRERUN:
            i = j - (DIVISIONS * PRERUN) - 1
            prop = 1 - (i / nr_preds)

            if i < nr_preds:
                try:
                    pred = predictions[i, :, :]
                    main_rgb.set_data(np.clip(rgb_image * prop, 0.0, 1.0))
                    main_pred.set_data(pred)
                    main_pred.set_alpha(np.clip(pred + (i / nr_preds * 1.333), 0.0, 1.0))

                    red_rgb.set_data(np.clip(red_square_rgb * prop, 0.0, 1.0))
                    red_square_pred = pred[sup1_rect[1]:sup1_rect[1] + size, sup1_rect[0]:sup1_rect[0] + size]
                    red_pred.set_data(red_square_pred)
                    red_pred.set_alpha(np.clip(red_square_pred + (i / nr_preds * 1.333), 0.0, 1.0))

                    blue_rgb.set_data(np.clip(blue_square_rgb * prop, 0.0, 1.0))
                    blue_square_pred = pred[sup2_rect[1]:sup2_rect[1] + size, sup2_rect[0]:sup2_rect[0] + size]
                    blue_pred.set_data(blue_square_pred)
                    blue_pred.set_alpha(np.clip(blue_square_pred + (i / nr_preds * 1.333), 0.0, 1.0))

                    time_text.set_text(str(round(times[j], 1)))
                except:
                    time_text.set_text(str(round(times[-1], 1)))
                    raise Exception("Animation Error")
            else:
                time_text.set_text(str(round(times[-1], 1)))
                print("Off by one error, ignoring")
        else:
            main_rgb.set_data(rgb_image)
            red_rgb.set_data(red_square_rgb)
            blue_rgb.set_data(blue_square_rgb)

        return [main_rgb, main_pred, red_rgb, red_pred, blue_rgb, blue_pred]

    anim = animation.FuncAnimation(fig, updatefig, frames=range(len(times)), interval=30, blit=True)
    anim.save(f"{FOLDER}animation-v5_{fig_name_abr}.mp4", writer=animation.FFMpegWriter(fps=30, bitrate=1000000), savefig_kwargs={"facecolor": IMG_BACKGROUND})
