import os
import sys

from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader

from src.exception import CustomException
from src.entity.config_entity import DataPreprocessingConfig
from src.logger import logging

class DataPreprocessing:

    def __init__(self):
        self.config : DataPreprocessingConfig = DataPreprocessingConfig()
    
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

    def create_loader(self, TRANSFORM_OBJ: transforms.Compose):
        '''
        The create_loaders method takes Transformations and create dataloaders.
        param TRANSFORM_OBJ:
        return: Dict of train, test, valid Loaders
        '''
        try:
            logging.info("creating data loader")
            result = {}

            train_data = ImageFolder(root=self.config.TRAIN_DATA_PATH, transform=TRANSFORM_OBJ)
            valid_data = ImageFolder(root=self.config.VALID_DATA_PATH, transform=TRANSFORM_OBJ)
            test_data = ImageFolder(root=self.config.TEST_DATA_PATH, transform=TRANSFORM_OBJ)

            logging.info("Created Image folder for each data type")

            train_dataloader = DataLoader(train_data, batch_size=self.config.BATCH_SIZE, shuffle=True)
            valid_dataloader = DataLoader(valid_data, batch_size=self.config.BATCH_SIZE, shuffle=False)
            test_dataloader = DataLoader(test_data, batch_size=self.config.BATCH_SIZE, shuffle=False)

            result = {
                "train_data_loader": (train_dataloader, train_data),
                "test_data_loader": (test_dataloader, test_data),
                "valid_data_loader": (valid_dataloader, valid_data)
            }
            logging.info("done creatign the dataloaders")
            return result
        except Exception as e:
            error = CustomException(e, sys)
            logging.error(error.error_message)
            raise error
    
    def run_step(self):
        try:
            TRANSFORM_OBJ = self.transformations()
            result = self.create_loader(TRANSFORM_OBJ)
            return result
        except Exception as e:
            error = CustomException(e, sys)
            logging.error(error.error_message)
            raise error
    
if __name__ == "__main__":
    data_preprocessing = DataPreprocessing()
    print(data_preprocessing.run_step())
            
