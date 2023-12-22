'''
Resample the train split of the dataset to balance the hair color propotion
Original: 0.4:0.2:0.2:0.1:0.1
To: 0.2,0.2,0.2,0.2,0.2

for the part bigger than average: random sample half of it
for the part smaller: use data agumentation to fill it

The original dataset structure:
data_face_imgs
├── anno.csv
└── images

The dataset structure after resample:
data
├── anno.csv
├── train
├── val
└── test
'''

import os, shutil
from tqdm import tqdm
import random
import numpy as np
import pandas as pd
from PIL import Image

from data_agumentation import enhance_brightness, random_rotate, \
    random_crop, random_rotate, enhance_contrast, random_flip, add_noise

def _read_anno(df, img_id):
        anno = {}
    
        head = df.columns.tolist()
        target_data = df[df.iloc[:, 0] == img_id].values.tolist()
        target_data = np.array(target_data)
        target_data[target_data=="-1"] = 0
        target_data[target_data=="1"] = 1
        
        for i in range(len(head)-1):
            anno[head[i+1]] = int(target_data[0][i+1])
            
        return anno

def double_img_list(orginial_list:list):
    doubled_img = orginial_list.copy()
    for img_name in tqdm(orginial_list):
        img_id = int(img_name.split(".")[0])
        agumented_img_name = str(img_id + 300000) + ".jpg"
        doubled_img.append(agumented_img_name)
    return doubled_img

def redouble_img_list(orginial_list:list):
    redoubled_img = orginial_list.copy()
    for img_name in tqdm(orginial_list):
        img_id = int(img_name.split(".")[0])
        agumented_img_name = str(img_id + 300000) + ".jpg"
        redoubled_img.append(agumented_img_name)
    for img_name in tqdm(orginial_list):
        img_id = int(img_name.split(".")[0])
        agumented_img_name = str(img_id + 600000) + ".jpg"
        redoubled_img.append(agumented_img_name)
    return redoubled_img
    
def main():
    workspace_dir = os.path.dirname(cur_dir)
    original_data_dir = os.path.join(workspace_dir,"data")
    output_data_dir = os.path.join(workspace_dir, "data_resampled")
    os.makedirs(output_data_dir, exist_ok=True)
    
    shutil.rmtree(output_data_dir)
    
    anno_path = os.path.join(original_data_dir,"anno.csv")
    df = pd.read_csv(anno_path)
    
    original_train_dir = os.path.join(original_data_dir,"train")
    original_train_list = os.listdir(original_train_dir)
    original_val_dir = os.path.join(original_data_dir,"val")
    original_val_list = os.listdir(original_val_dir)
    original_test_dir = os.path.join(original_data_dir,"test")
    original_test_list = os.listdir(original_test_dir)
    
    black_hair_img = []
    blond_hair_img = []
    brown_hair_img = []
    gray_hair_img = []
    others_img = []
    
    for img_id in tqdm(original_train_list):
        anno = _read_anno(df, img_id)
        if anno["Black_Hair"] == 1:
            black_hair_img.append(img_id)
        elif anno["Blond_Hair"] == 1:
            blond_hair_img.append(img_id)
        elif anno["Brown_Hair"] == 1:
            brown_hair_img.append(img_id)
        elif anno["Gray_Hair"] == 1:
            gray_hair_img.append(img_id)
        else:
            others_img.append(img_id)
    
    # black_hair_img = random.sample(black_hair_img,len(black_hair_img)//2)
    # brown_hair_img = double_img_list(brown_hair_img)
    # gray_hair_img = double_img_list(gray_hair_img)
    others_img = double_img_list(others_img)
    resampled_train_list = black_hair_img + blond_hair_img + \
        brown_hair_img + gray_hair_img + others_img
    
    random.shuffle(resampled_train_list)
    dataset_split = {
        "train": resampled_train_list,
        "val": original_val_list,
        "test": original_test_list
    }
    
    for split, imgs in dataset_split.items():
        output_path = os.path.join(output_data_dir, split)
        os.makedirs(output_path, exist_ok=True)
        for img in tqdm(imgs):
            img_id = int(img.split(".")[0])
            img_path = os.path.join(original_data_dir, split)
            img_after_path = os.path.join(output_data_dir, split, img)
            if img_id < 300000:
                shutil.copy2(os.path.join(img_path, img), img_after_path)
            else:
                # Use data agumentation to double the dataset
                if img_id < 600000:
                    img = Image.open(os.path.join(img_path, str(img_id-300000).zfill(6)+".jpg"))
                else:
                    img = Image.open(os.path.join(img_path, str(img_id-600000).zfill(6)+".jpg"))

                # Randomly select a data augmentation operation
                augmentation = random.choice(["noise", "contrast", "flip","brightness", "crop", "rotate"])

                # Apply the selected data augmentation operation
                if augmentation == "noise":
                    img = add_noise(img)
                elif augmentation == "contrast":
                    img = enhance_contrast(img)
                elif augmentation == "flip":
                    img = random_flip(img)
                elif augmentation == "brightness":
                    img = enhance_brightness(img)
                elif augmentation == "crop":
                    img = random_crop(img)
                elif augmentation == "rotate":
                    img = random_rotate(img)
                img.save(img_after_path)
            
    shutil.copy2(os.path.join(original_data_dir,"anno.csv"), os.path.join(output_data_dir,"anno.csv"))
    
if __name__ == "__main__":
    cur_dir = os.path.dirname(__file__)
    main()