import os, sys
from typing import Dict

import boto3
from dotenv import load_dotenv

from src.utils.utils import unique_image_name
from src.exeception import CustomException
from src.logger import logging


load_dotenv()
class S3Connection:
    '''
    Class used to make and manage connection with S3 bucket
    '''
    def __init__(self):
        self.session = boto3.Session(
            aws_access_key_id = os.environ['AWS_ACCESS_KEY_ID'],
            aws_secret_access_key = os.environ['AWS_SECRET_ACCESS_KEY']
        )
        self.s3 = self.session.resource("s3")
        self.bucket = self.s3.Bucket(os.environ['AWS_BUCKET_NAME'])
    


