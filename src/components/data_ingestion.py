import os
import splitfolders
import sys

from src.exception import CustomException
from src.logger import logging
from src.entity.config_entity import DataIngestionConfig
from from_root import from_root
from src.utils.storage_handler import S3Connector


class DataIngestion:
    def __init__(self):
        self.config: DataIngestionConfig = DataIngestionConfig()
    

    def download_dir(self):
        try:
            logging.info("about tho download data")
            data_path = os.path.join(from_root(), self.config.RAW, self.config.PREFIX)
            os.system(f"aws s3 sync {self.config.URI} {data_path}")
            logging.info("done downloading the data")
        except Exception as e:
            error = CustomException(e, sys)
            logging.error(error.error_message)
            raise error
    
    def split_data(self):
        '''
        the method split the data for which allows for easy training
        '''
        try:
            splitfolders.ratio(
                input=os.path.join(self.config.RAW, self.config.PREFIX),
                output=self.config.SPILT,
                seed=self.config.SEED,
                ratio=self.config.RATIO,
                group_prefix=None,
                move=False
            )
        except Exception as e:
            error = CustomException(e, sys)
            logging.error(error.error_message)
            raise error
    
    def run_step(self):
        try:
            self.download_dir()
            self.split_data()
        except Exception as e:
            error = CustomException(e, sys)
            logging.error(error.error_message)
            raise error

if __name__ == "__main__":
    paths = ["data", r"data/raw", r"data/split", r"data/embeddings",
             "model", r"model/benchmark", r"model/fintued"]
    for folder in paths:
        path = os.path.join(from_root(), folder)
        if not os.path.exists(path):
            os.mkdir(path)
    
    data_ingestion = DataIngestion()
    data_ingestion.run_step()
