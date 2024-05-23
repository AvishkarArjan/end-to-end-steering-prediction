from PIL import Image, ImageFilter, ImageEnhance, ImageFile
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import time
import shutil

ImageFile.LOAD_TRUNCATED_IMAGES = True

def crop(img):
    width, height = img.size
    img = img.crop((0, 120, width, height))
    return img


def blur(img):
    img = img.filter(ImageFilter.BLUR)
    return img

def darken(img):
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(0.5) # value proportional to brightness
    return img

def preprocess(data_path, save_dir):
    start = time.time()
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
        print("Save Dir created")
    

    for file in data_path.iterdir():
        if str(file.suffix) == ".jpg":
            img = Image.open(data_path/file)
            img = crop(img)
            img = blur(img)
            img = darken(img)

            print(f"Saving : {save_dir/file.name}")
            img.save(save_dir/file.name)

        if str(file.stem)=="data":
            shutil.copy(str(file),str(save_dir/file.name) )

    end = time.time()

    print(f"Total Processing tim : {end-start}s")
            

def vis_image(img):
    # plt.imshow(np.transpose(img,  (1, 2, 0)))
    plt.imshow(img)
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    root_dir = Path("/home/avishkar/Desktop/research")
    data_path = root_dir/"driving_dataset"
    save_dir = root_dir/"driving_dataset_preprocessed"

    preprocess(data_path=data_path, save_dir=save_dir)


