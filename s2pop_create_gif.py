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
FADE = 3
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
    preruns = [rgb_image] * DIVISIONS * PRERUN
    
    interpolated = np.linspace(rgb_image, np.zeros_like(rgb_image), DIVISIONS * FADE, axis=0)
    for q in range(DIVISIONS * FADE):
        preruns.append(interpolated[q, :, :])
    interpolated = None
    
    im1 = ax.imshow(rgb_image, vmin=0.0, vmax=1.0, interpolation="antialiased")
    times = np.arange(0, LIMIT, 1 / DIVISIONS)

    time_text = ax.text(0.05, 0.05, str(round(times[0], 1)), fontsize=15, color=TEXT_COLOR, transform=ax.transAxes)
    place_text = ax.text(0.5, 0.95, places[idx]["name"], fontsize=15, color=TEXT_COLOR, transform=ax.transAxes)

    count_init = 0
    count_pred = 0
    def updatefig(j):
        global count_init, count_pred

        try:
            if count_init <= len(preruns):
                im1.set_data(preruns[j])
                count_init += 1
            else:
                im1.set_cmap(IMG_RAMP)
                im1.set_clim(vmin=0.0, vmax=1.0)
                im1.set_data(img1[count_pred])
                time_text.set_text(str(round(times[count_pred], 1)))
                count_pred += 1
        except:
            time_text.set_text(str(len(times)))
            pass

        return [im1, time_text]

    anim = animation.FuncAnimation(fig, updatefig, frames=range(len(img1) + len(preruns)), interval=30, blit=True)
    anim.save(f"/home/casper/Desktop/UNICEF/animation-v31_{places[idx]['name_abr']}.mp4", writer=animation.FFMpegWriter(fps=30, bitrate=1000000))
