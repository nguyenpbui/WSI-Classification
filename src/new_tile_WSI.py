import os
import time
import pyvips
import numpy as np
from PIL import Image

os.environ['VIPS_DISC_THRESHOLD'] = '9gb'

def extract_image_tiles(p_img, base_folder, images_id, size: int = 768, 
                        scale: float = 1.0, drop_thr: float = 0.8) -> list:
    
    name, _ = os.path.splitext(os.path.basename(p_img))
    folder = os.path.join(base_folder, f"{images_id}_files/20.0")
    os.makedirs(folder, exist_ok=True)
    
    im = pyvips.Image.new_from_file(p_img)
    w = h = size
#     new_w, new_h = w//4, h//4
#     im = pyvips.Image.thumbnail(p_img, (new_w, new_h))
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
        
        col_id = int(x_ / w)
        row_id = int(y_ / h)
        p_img = os.path.join(folder, f"{col_id}-{row_id}.png")
        # print(tile.shape, tile.dtype, tile.min(), tile.max())
        new_size = int(size * scale), int(size * scale)
        Image.fromarray(tile).resize(new_size, Image.LANCZOS).save(p_img)
        files.append(p_img)
        idxs.append(idx)
    return files, idxs

if __name__ == '__main__':
    DATASET_IMAGES = "../data/UBC-OCEAN/test_images"
    time1 = time.time()

    tiles_img, ids = extract_image_tiles(os.path.join(DATASET_IMAGES, "41.png"),
        "../data/Tiled_UBC", "512", size=2048, scale=0.25)

    print(time.time() - time1)