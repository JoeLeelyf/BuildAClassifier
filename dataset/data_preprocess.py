'''
Split the dataset into train/val and test set

The original dataset structure:
data_face_imgs
├── anno.csv
└── images

The dataset structure after preprocess:
data
├── anno.csv
├── train
├── val
└── test
'''

import os, shutil
from tqdm import tqdm
import random

def main():
    workspace_dir = os.path.dirname(cur_dir)
    original_data_dir = os.path.join(workspace_dir,"data_face_imgs")
    output_data_dir = os.path.join(workspace_dir, "data")
    os.makedirs(output_data_dir, exist_ok=True)
    
    shutil.rmtree(output_data_dir)
    
    original_img_dir = os.path.join(original_data_dir, "images")
    
    original_img_list = os.listdir(original_img_dir)
    random.shuffle(original_img_list)
    dataset_length = len(original_img_list)
    
    portion = 1
    dataset_split = {
        "train": original_img_list[0 : int(portion * 0.8 * dataset_length)],
        "val": original_img_list[int(portion * 0.8 * dataset_length) : int(portion * 0.9 * dataset_length)],
        "test": original_img_list[int(portion * 0.9 * dataset_length) : int(portion * dataset_length)]
    }
    
    for split, imgs in dataset_split.items():
        output_path = os.path.join(output_data_dir, split)
        os.makedirs(output_path, exist_ok=True)
        for img in tqdm(imgs):
            img_path = os.path.join(original_img_dir, img)
            img_after_path = os.path.join(output_data_dir, split, img)
            shutil.copy2(img_path, img_after_path)
            
    shutil.copy2(os.path.join(original_data_dir,"anno.csv"), os.path.join(output_data_dir,"anno.csv"))
    
if __name__ == "__main__":
    cur_dir = os.path.dirname(__file__)
    main()