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
            full_path = os.path.join(self.config.ROOT_DIR, class_path)
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
        images, label, link = record.img, record.label, record.s3link
        images = Image.open(images)

        if len(images.getbands()) < 3:
            images = images.convert('RGB')
        
        images = np.array(self.tranform(images))
        label = torch.from_numpy(np.array(label))
        images = torch.from_numpy(images)

        return images, label, link


class EmbeddingGenerator:

    def __init__(self, model: torch.nn.Module, device: str):
        self.config: EmbeddingConfig = EmbeddingConfig()
        self.mongo: MongoDBClient = MongoDBClient()
        self.model: torch.nn.Module = model
        self.device = device
        self.emebedding_model:torch.nn.Sequential = self.load_model()
        self.emebedding_model.eval()


    def load_model(self):
        model = self.model.to(self.device)
        model.load_state_dict(torch.load(self.config.PATH, map_location=self.device))
        return nn.Sequential(*list(model.children())[:-1])

    def run_step(self, batch_num, images, labels, s3_links):

        try:
            logging.info("about to start the process in Emebedding Generator")

            record = dict()

            # getting the embeddings
            images = self.emebedding_model(images.to(self.device))
            images = images.detach().cpu().numpy()

            record["images"] = images.tolist()
            record["labels"] = labels.tolist()
            record["s3_links"] = s3_links
            
            # Creating DF with image label and s3_links
            df = pd.DataFrame(record)

            # inserting the info to mongo
            records = list(json.loads(df.T.to_json()).values())
            self.mongo.insert_bulk_records(records)

            logging.info(f"operation complete for batch {batch_num}")

        except Exception as e:
            error = CustomException(e, sys)
            logging.error(error.error_message)
            raise error

if __name__ == "__main__":
    dp = DataPreprocessing()
    loaders = dp.run_step()
    model = NeuralNet()
    # just a small sample to test thing out
    for name, val in loaders.items():         
        data = ImageFolder(label_map=val[1].class_to_idx)
        dataloader = DataLoader(dataset=data, batch_size=64, shuffle=True)

        embeds = EmbeddingGenerator(model=model, device="mps")
        for batch_num, values in tqdm(enumerate(dataloader)):
            img, target, link = values
            embeds.run_step(batch_num, img, target, link)

