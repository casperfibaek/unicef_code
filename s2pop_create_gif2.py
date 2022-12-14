import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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

def ready_image(glob_path, divisions=5, limit=100):
    images = glob(glob_path)
    images.sort(key=cmp_to_key(cmp_items))

    arrays = []
    for img in images[:limit]:
        img = beo.raster_to_array(img)[:, :, 0]
        arrays.append(img)

    arrays_interpolated = []
    for idx, arr in enumerate(arrays):
        if idx == 0:
            arrays_interpolated.append(arr)
        else:
            interpolated = np.linspace(arrays[idx-1], arr, divisions, axis=0)
            for i in range(divisions):
                arrays_interpolated.append(interpolated[i, :, :])

    return arrays_interpolated

DIVISIONS = 10
DPI = 130
HEIGHT = 1000
WIDTH = 1000
LIMIT = 101
PRERUN = 3
TEXT_COLOR = "#BBBBBB"
IMG_RAMP = "magma"

places = [
    { "name": "San Diego", "name_abr": "san-diego" },
    { "name": "Palm Springs", "name_abr": "palm-springs" },
    { "name": "Lake Isabella", "name_abr": "lake-isabella" },
    { "name": "Nevada Border", "name_abr": "nevada-border" },
    { "name": "Dixon", "name_abr": "dixon" },
    { "name": "Lake Port", "name_abr": "lake-port" },
    { "name": "Eugene", "name_abr": "eugene" },
    { "name": "Covelo", "name_abr": "covelo" },
    { "name": "Homestead", "name_abr": "homestead" },
    { "name": "Melbourne", "name_abr": "melbourne" },
]

for idx, fig_nr in enumerate(["0", "10", "20", "30", "40", "50", "60", "70", "80", "90"]):
    if fig_nr == "0" or fig_nr == "10":
        continue

    fig = plt.figure(figsize=(HEIGHT / DPI, WIDTH / DPI), dpi=DPI, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    rgb_image = f"/home/casper/Desktop/UNICEF/gifs/test_images/north-america_{fig_nr}.tif"
    rgb_image = beo.raster_to_array(rgb_image)
    rgb_image = rgb_image[:, :, 1:4] / 10000.0

    q02 = np.quantile(rgb_image, 0.02)
    q99 = np.quantile(rgb_image, 0.99)

    rgb_image = np.clip(rgb_image[:, :, ::-1], q02, q99)
    rgb_image = (rgb_image - q02) / (q99 - q02)

    img1 = ready_image(f"/home/casper/Desktop/UNICEF/gifs/s2pop_gan_v20_north-america_{fig_nr}_*.tif", divisions=DIVISIONS, limit=LIMIT)
    
    im1 = ax.imshow(rgb_image, vmin=0.0, vmax=1.0, interpolation="antialiased")
    im2 = ax.imshow(img1[0], vmin=0.0, vmax=1.0, interpolation="antialiased", cmap=IMG_RAMP, alpha=np.zeros_like(img1[0]))

    times = [0.0] * (DIVISIONS * PRERUN) + list(np.arange(0, LIMIT, 1 / DIVISIONS))

    time_text = ax.text(0.05, 0.05, str(round(times[0], 1)), fontsize=15, color=TEXT_COLOR, transform=ax.transAxes)
    place_text = ax.text(0.5, 0.95, places[idx]["name"], fontsize=15, color=TEXT_COLOR, transform=ax.transAxes)

    def updatefig(j):
        global im1, im2

        im1.set_data(rgb_image)

        if j >= DIVISIONS * PRERUN:
            i = j - (DIVISIONS * PRERUN)
            prop = 1 - (i / len(img1))
            try:
                im1.set_data(np.clip(rgb_image * prop, 0.0, 1.0))

                im2.set_data(img1[i])
                im2.set_alpha(np.clip(img1[i] + (i / len(img1) * 1.333), 0.0, 1.0))
                time_text.set_text(str(round(times[j], 1)))
            except:
                time_text.set_text(str(round(times[-1], 1)))
                pass

        return [im1, im2]

    anim = animation.FuncAnimation(fig, updatefig, frames=range(len(times)), interval=30, blit=True)
    anim.save(f"/home/casper/Desktop/UNICEF/animation-v32_{places[idx]['name_abr']}.mp4", writer=animation.FFMpegWriter(fps=30, bitrate=1000000))
