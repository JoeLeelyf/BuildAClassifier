import os
from tqdm import tqdm

import pandas as pd
import numpy as np
from PIL import Image

from torch.utils.data import Dataset
from torchvision.transforms import transforms

class ClassifierDataSet(Dataset):
    def __init__(self, args, split) -> None:
        self.args = args
        self.root_dir = args.data_dir
        self.split = split
        self.img_dir = os.path.join(self.root_dir, split)
        self.anno_path = os.path.join(self.root_dir, "anno.csv")
        
        self.samples = self._create_dataset()
        self.transform = transforms.Compose([
            transforms.Resize(size=224, max_size=None, antialias=None),
            transforms.CenterCrop(size=(224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])

    def __len__(self):
        return len(self.samples)
        
    def _read_anno(self, df, img_id):
        img_num = int(img_id.split('.')[0])
        if img_num >= 300000 and img_num <= 600000:
            img_num -= 300000
            img_id = str(img_num).zfill(6) + ".jpg"
        elif img_num >= 600000:
            img_num -= 600000
            img_id = str(img_num).zfill(6) + ".jpg"
        anno = {}
    
        head = df.columns.tolist()
        target_data = df[df.iloc[:, 0] == img_id].values.tolist()
        target_data = np.array(target_data)
        target_data[target_data=="-1"] = 0
        target_data[target_data=="1"] = 1
        
        for i in range(len(head)-1):
            anno[head[i+1]] = int(target_data[0][i+1])
            
        return anno
    
    def _create_dataset(self):
        dataset = []
        print("Creating dataset %s ..."%self.split)
        
        df = pd.read_csv(self.anno_path)
        
        for img_id in tqdm(os.listdir(self.img_dir)):
            img_path = os.path.join(self.img_dir, img_id)
            anno = self._read_anno(df, img_id)
            
            sample = {}
            sample["img_path"] = img_path
            sample["anno"] = anno
            
            dataset.append(sample)
            
        print("len of dataset: %f"%len(dataset))
        return dataset
    
    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        img_path, anno = (
            sample["img_path"],
            sample["anno"]
        )
        img = Image.open(img_path)    
        img = self.transform(img)
        
        return_dict = {
            "img": img,
            "anno": anno
        }
        
        return return_dict
    