import os
import pyvips
import numpy as np
from PIL import Image
import time
import glob

os.environ['VIPS_DISC_THRESHOLD'] = '9gb'

def extract_image_tiles(
    p_img, base_folder, images_id, size: int = 768, scale: float = 1.0, drop_thr: float = 0.8
) -> list:
    name, _ = os.path.splitext(os.path.basename(p_img))
    
    folder = os.path.join(base_folder, f"{images_id}_files/20.0")
    os.makedirs(folder, exist_ok=True)
    
    im = pyvips.Image.new_from_file(p_img)
    w = h = size
    print("Before resize: ", im.width, im.height)

    if im.width > 4000 and im.height > 4000:
        # For WSI
        im = im.resize(0.4)
    else:
        # For TMA
        im = im

    print("After resize: ", im.width, im.height)

    # https://stackoverflow.com/a/47581978/4521646
    inds = [(y, y + h, x, x + w)
            for y in range(0, im.height, h) for x in range(0, im.width, w)]
    files, idxs = [], []
    for k, idx in enumerate(inds):
        y, y_, x, x_ = idx
        # https://libvips.github.io/pyvips/vimage.html#pyvips.Image.crop
        tile = im.crop(x, y, min(w, im.width - x), min(h, im.height - y)).numpy()[..., :3]
        if tile.shape[:2] != (h, w):
            tile_ = tile
            tile_size = (h, w) if tile.ndim == 2 else (h, w, tile.shape[2])
            tile = np.zeros(tile_size, dtype=tile.dtype)
            tile[:tile_.shape[0], :tile_.shape[1], ...] = tile_
        mask_bg = np.sum(tile, axis=2) == 0
        if np.sum(mask_bg) >= (np.prod(mask_bg.shape) * drop_thr):
            #print(f"skip almost empty tile: {k:06}_{int(x_ / w)}-{int(y_ / h)}")
            continue
        tile[mask_bg, :] = 255
        mask_bg = np.mean(tile, axis=2) > 240
        if np.sum(mask_bg) >= (np.prod(mask_bg.shape) * drop_thr):
            #print(f"skip almost empty tile: {k:06}_{int(x_ / w)}-{int(y_ / h)}")
            continue
    #         p_img = os.path.join(folder, f"{k:06}_{int(x_ / w)}-{int(y_ / h)}.png")
        
        col_id = int(x_ / w) - 1
        row_id = int(y_ / h) - 1
        p_img = os.path.join(folder, f"{col_id}-{row_id}.png")
        # print(tile.shape, tile.dtype, tile.min(), tile.max())
        new_size = int(size * scale), int(size * scale)
        Image.fromarray(tile).resize(new_size, Image.LANCZOS).save(p_img)
        files.append(p_img)
        idxs.append(idx)
    
    return files, idxs

DATASET_IMAGES = "./data/UBC-OCEAN/train_images/*.png"
save_folder = "./data/pyvip_tiling_selective"
time0 = time.time()

for img_path in glob.glob(DATASET_IMAGES):
    time1 = time.time()

    images_id = os.path.splitext(os.path.basename(img_path))[0]
    print(f"Processing {img_path}...")  

    tiles_img, ids = extract_image_tiles(img_path, save_folder, images_id, size=512, scale=1)
    print(time.time()-time1)

print("Total time:", time.time()-time0)
