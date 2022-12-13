import os
import numpy as np
import pandas as pd
import torch 
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision.io import read_image
from PIL import Image
import torchvision.transforms as T

def from_class_to_yolo_cat(class_name) -> torch.int8:
            if class_name == "triton":
                return 1
            elif class_name == "grenouille-crapaud":
                return 2
            elif class_name == "planche":
                return 3
            elif class_name == "feuille":
                return 4
            elif class_name == "souris":
                return 5
            elif class_name == "insecte":
                return 6
            else:
                return -1 # error


class CrapaudSet(Dataset):
    def __preproccessing(self, df_annotation : pd.DataFrame) -> pd.DataFrame:
        
        df_annotation['x1'] = df_annotation['top_left_x']
        df_annotation['y1'] = df_annotation['top_left_y'] 
        df_annotation['x2'] = df_annotation['top_left_x'] + df_annotation['w']
        df_annotation['y2'] = df_annotation['top_left_y'] + df_annotation['h']
        df_annotation['class'] = df_annotation['class'].apply(from_class_to_yolo_cat)
        return df_annotation

    def __init__(self, annotations_file : str, img_dir : str, transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.img_labels = self.__preproccessing(self.img_labels)

    def __len__(self):
        return len(self.img_labels['path'].unique())

   
    def __getitem__(self, idx):
        # We can have multiple labels for one image therefore we take the unique paths 
        # and get all the labels associated with it
        
        # get the image path
        path = (self.img_labels['path'].unique()[idx])
        img_path = os.path.join(self.img_dir, path)
        img = Image.open(img_path).convert("RGB")
        
        # select all the labels for the image
        labels = self.img_labels[self.img_labels['path'] == path]
        # get the bounding boxe(S), there might be multiple !!
        boxes_np = labels[['x1', 'y1', 'x2', 'y2']].values
        labels_np = labels['class'].values
        # create the target dict 
        target = {}
        target['boxes'] = torch.as_tensor(boxes_np, dtype=torch.float32)
        target['labels'] = torch.as_tensor(labels_np, dtype=torch.int16)
        target['image_id'] = torch.tensor([idx])
        target['iscrowd'] = torch.zeros((len(boxes_np),), dtype=torch.int64)
        if self.transform is not None:
            img = self.transform(img)
        return img, target