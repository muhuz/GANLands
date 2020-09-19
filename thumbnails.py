import numpy as np
import os
from PIL import Image

path = 'land_imgs/full/'
save_path = 'land_imgs/thumbnail/'
filenames = os.listdir(path)

for name in filenames:
    img = Image.open(os.path.join(path, name))
    img_array = np.array(img)
    # remove the bottom 10% of image to delete watermarks
    img_height = img_array.shape[0]
    cropped_img_array = img_array[:-(img_height // 14),:,:]
    small_dim = np.argmin(cropped_img_array.shape[:2])
    size = min(cropped_img_array.shape[:2])
    # height is smaller
    if small_dim == 0:
        # get middle segement
        width = cropped_img_array.shape[1]
        start = (width - size) // 2
        square_img = cropped_img_array[:, start:start+size, :] 
    else:
        # get middle segement
        height  = cropped_img_array.shape[0]
        start = (height - size) // 2
        square_img = cropped_img_array[start:start+size, :, :] 

    final_img = Image.fromarray(square_img, 'RGB')
    final_img.thumbnail((256,256))
    final_img.save(os.path.join(save_path, name))