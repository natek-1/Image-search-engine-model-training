import os, sys, json
from typing import Dict, List
from collections import namedtuple
from PIL import Image
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
import numpy as np

from src.entity.config_entity import EmbeddingConfig, ImageFolderConfig
from src.components.data_preprocessing import DataPreprocessing
from src.components.model import NeuralNet
from src.utils.database_handler import MongoDBClient
from src.exception import CustomException
from src.logger import logging


ImageRecord = namedtuple("ImageRecord", ["img", "label", "s3link"])


class ImageFolder(Dataset):

    def __init__(self, label_map: Dict):
        self.config: ImageFolderConfig = ImageFolderConfig()
        self.config.LABEL_MAP = label_map
        self.tranform = self.transformations()
        self.image_record: List[ImageRecord] = []
        self.record = ImageRecord
        list_dir = os.listdir(self.config.ROOT_DIR)

        for class_path in list_dir:
            full_path = os.path.join(list_dir, class_path)
            images = os.listdir(full_path)
            for image in images:
                image_path = Path(f"{self.config.ROOT_DIR}/{class_path}/{image}")
                self.image_record.append(self.record(
                    img = image_path,
                    label= self.config.LABEL_MAP[class_path],
                    s3link=self.config.S3_LINK.format(self.config.BUCKET, class_path, image)
                ))


    def transformations(self) -> transforms.Compose:
        '''
        Transformation Method Provides transforms.Compose object. Its pytorch's transformation class to apply on images.
        return: transforms.Compose object
        '''
        TRANSFORM_OBJ = transforms.Compose(
            [
                transforms.Resize(self.config.IMAGE_SIZE),
                transforms.CenterCrop(self.config.IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.config.MEAN,
                                    std=self.config.STD)
            ]
        )

        return TRANSFORM_OBJ

    def __len__(self):
        return len(self.image_record)
    

    def __getitem__(self, index):
        record = self.image_record[index]
        image, label, link = record.image, record.label, record.s3link
        image = Image.open(image)

        if len(images.getbands()) < 3:
            images = images.convert('RGB')
        
        image = np.array(self.tranform(image))
        label = torch.from_numpy(np.array(label))
        image = torch.from_numpy(image)

        return image, label, link

