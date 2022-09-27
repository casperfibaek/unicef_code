import os
from glob import glob
from s2super import super_sample, get_s2super_model
import buteo as beo

model_super = get_s2super_model()
folder = "F:/UNICEF/eq_guinea_images/"

images = glob(folder + "*_*.tif")
for idx, image_path in enumerate(images):
    if "label" in image_path:
        continue

    name = os.path.splitext(os.path.basename(image_path))[0]

    read_image = beo.raster_to_array(image_path)
    super_sampled = super_sample(read_image[:, :, 1:], method="fast", fit_data=True, preloaded_model=model_super, iterations=1)
    read_image[:, :, 1:] = super_sampled
    beo.array_to_raster(read_image, reference=image_path, out_path=folder + name + "_i1.tif")

    read_image = beo.raster_to_array(image_path)
    super_sampled = super_sample(read_image[:, :, 1:], method="fast", fit_data=True, preloaded_model=model_super, iterations=2)
    read_image[:, :, 1:] = super_sampled
    beo.array_to_raster(read_image, reference=image_path, out_path=folder + name + "_i2.tif")

    read_image = beo.raster_to_array(image_path)
    super_sampled = super_sample(read_image[:, :, 1:], method="fast", fit_data=True, preloaded_model=model_super, iterations=3)
    read_image[:, :, 1:] = super_sampled
    beo.array_to_raster(read_image, reference=image_path, out_path=folder + name + "_i3.tif")

    print(f"Processed: {idx + 1} of {len(images)}")
