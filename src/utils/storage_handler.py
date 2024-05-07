import os, sys

import boto3
from dotenv import load_dotenv
import tarfile

from src.entity.config_entity import s3Config
from src.utils.common import get_unique_filename
from src.exception import CustomException
from src.logger import logging


load_dotenv()
class S3Connector:
    '''
    Class used to make and manage connection with S3 bucket
    '''
    def __init__(self):
        self.config: s3Config = s3Config()
        self.session = boto3.Session(
            aws_access_key_id = self.config.ACCESS_KEY,
            aws_secret_access_key = self.config.SECRET_KEY,
            region_name=self.config.REGION
        )
        self.s3 = self.session.resource("s3")
        self.client = self.session.client("s3")

        self.bucket = self.s3.Bucket(self.config.BUCKET)
    
    def zip_file(self):
        folder = tarfile.open(self.config.ZIP_NAME, "w:gz")
        for path, name in self.config.ZIP_PATHS:
            folder.add(path, name)
        self.s3.meta.client.upload_file(self.config.ZIP_NAME, self.config.BUCKET,
                                        f'{self.config.KEY}/{self.config.ZIP_NAME}')
        os.remove(self.config.ZIP_NAME)
    
    def pull_artifacts(self):
        self.bucket.download_file(f'{self.config.KEY}/{self.config.ZIP_NAME}', self.config.ZIP_NAME)
        folder = tarfile.open(self.config.ZIP_NAME)
        folder.extractall()
        folder.close()
        os.remove(self.config.ZIP_NAME)

if __name__ == "__main__":
    connection = S3Connector()
    connection.pull_artifacts()
    


