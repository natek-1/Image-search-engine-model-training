from from_root import from_root
from dotenv import load_dotenv
from typing import Tuple
import os

load_dotenv()

class DataBaseConfig:
    def __init__(self):
        self.USERNAME: str = os.environ['ATLAS_CLUSTER_USERNAME']
        self.PASSWORD: str = os.environ['ATLAS_CLUSTER_PASSWORD']
        self.URL: str = f"mongodb+srv://{self.USERNAME}:{self.PASSWORD}@learn.fz36e1j.mongodb.net/?retryWrites=true&w=majority&appName=Learn"
        self.DATABASENAME: str = os.environ["DATABASE_NAME"]
        self.COLLECTION: str = "Embeddings"
    
    def get_database_config(self):
        return self.__dict__


class s3Config:
    def __init__(self):
        self.ACCESS_KEY = os.environ['AWS_ACCESS_KEY_ID']
        self.SECRET_KEY = os.environ['AWS_SECRET_ACCESS_KEY']
        self.REGION = os.environ['AWS_REGION']
        self.BUCKET = os.environ['AWS_BUCKET_NAME']
        self.KEY = "model"
        self.ZIP_NAME = "data.zip"
        self.ZIP_PATHS = [(os.path.join(from_root(), "data", "embeddings", "embeddings.json"), "embeddings.json"),
                          (os.path.join(from_root(), "data", "embeddings", "embeddings.ann"), "embeddings.ann"),
                          (os.path.join(from_root(), "model", "finetuned", "model.pth"), "model.pth")]
        
    def get_s3config(self):
        return self.__dict__


class DataIngestionConfig:
    def __init__(self):
        self.PREFIX: str = "images"
        self.RAW: str = "data/raw"
        self.SPILT: str= "data/split"
        self.BUCKET: str = os.environ['AWS_BUCKET_NAME']
        self.SEED: int = 42
        self.RATIO: Tuple = (0.8, 0.1, 0.1)
        self.URI = os.environ["AWS_S3_URI"]
    
    def get_data_ingestion_config(self):
        return self.__dict__

class DataPreprocessingConfig:
    def __init__(self):
        self.BATCH_SIZE = 32
        self.IMAGE_SIZE = 256
        self.TRAIN_DATA_PATH = os.path.join(from_root(), "data", "split", "train")
        self.TEST_DATA_PATH = os.path.join(from_root(), "data", "split", "test")
        self.VALID_DATA_PATH = os.path.join(from_root(), "data", "split", "val")
        self.MEAN = [0.485, 0.456, 0.406]
        self.STD = [0.229, 0.224, 0.225]
    
    def get_data_preprocessing_config(self):
        return self.__dict__