import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'frame-interpolation'))

import json
import logging
import tempfile

import boto3
import numpy as np
import tensorflow as tf

from eval.interpolator import Interpolator
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

AWS_REGION = os.environ['AWS_REGION']
SQS_QUEUE_URL = os.environ['SQS_QUEUE_URL']
S3_BUCKET = os.environ['S3_BUCKET']
MODEL_PATH = os.environ['MODEL_PATH']
SES_SENDER_EMAIL = os.environ['SES_SENDER_EMAIL']
PRESIGNED_URL_EXPIRY_SECONDS = int(os.environ.get('PRESIGNED_URL_EXPIRY_SECONDS', 604800))


class InterpolationWorker:
    def __init__(self):
        self.sqs = boto3.client('sqs', region_name=AWS_REGION)
        self.s3 = boto3.client('s3', region_name=AWS_REGION)
        self.ses = boto3.client('ses', region_name=AWS_REGION)
        logger.info("Loading FILM model from %s", MODEL_PATH)
        self.interpolator = Interpolator(MODEL_PATH, align=64, block_shape=[1, 1])
        logger.info("Model loaded.")

    def _read_image(self, path):
        image_data = tf.io.read_file(path)
        image = tf.io.decode_image(image_data, channels=3)
        return tf.cast(image, tf.float32).numpy() / 255.0

    def _write_image(self, path, image):
        uint8 = (np.clip(image * 255.0, 0, 255) + 0.5).astype(np.uint8)
        tf.io.write_file(path, tf.io.encode_png(uint8))

    def process_message(self, message):
        body = json.loads(message['Body'])
        job_id = body['JobId']
        email = body['Email']
        start_key = body['StartFrameKey']
        end_key = body['EndFrameKey']

        logger.info("Processing job %s for %s", job_id, email)

        with tempfile.TemporaryDirectory() as tmpdir:
            start_path = os.path.join(tmpdir, 'start.png')
            end_path = os.path.join(tmpdir, 'end.png')
            mid_path = os.path.join(tmpdir, 'middle.png')

            logger.info("Downloading %s", start_key)
            self.s3.download_file(S3_BUCKET, start_key, start_path)
            logger.info("Downloading %s", end_key)
            self.s3.download_file(S3_BUCKET, end_key, end_path)

            img1 = self._read_image(start_path)
            img2 = self._read_image(end_path)

            img1_batch = np.expand_dims(img1, 0)
            img2_batch = np.expand_dims(img2, 0)
            mid = self.interpolator(img1_batch, img2_batch, np.array([0.5], dtype=np.float32))[0]

            self._write_image(mid_path, mid)

            output_key = os.path.dirname(start_key) + '/middle.png'
            logger.info("Uploading result to %s", output_key)
            self.s3.upload_file(mid_path, S3_BUCKET, output_key)

        presigned_url = self.s3.generate_presigned_url(
            'get_object',
            Params={'Bucket': S3_BUCKET, 'Key': output_key},
            ExpiresIn=PRESIGNED_URL_EXPIRY_SECONDS,
        )

        logger.info("Sending email to %s", email)
        self.ses.send_email(
            Source=SES_SENDER_EMAIL,
            Destination={'ToAddresses': [email]},
            Message={
                'Subject': {'Data': f'Your interpolated frame is ready (Job {job_id})'},
                'Body': {
                    'Text': {
                        'Data': (
                            f'Your frame interpolation job has completed.\n\n'
                            f'Download your middle frame here (link valid for 7 days):\n'
                            f'{presigned_url}\n\n'
                            f'Job ID: {job_id}\n'
                        )
                    }
                },
            },
        )

        self.sqs.delete_message(
            QueueUrl=SQS_QUEUE_URL,
            ReceiptHandle=message['ReceiptHandle'],
        )
        logger.info("Job %s complete.", job_id)

    def run(self):
        logger.info("Worker started. Polling %s", SQS_QUEUE_URL)
        while True:
            resp = self.sqs.receive_message(
                QueueUrl=SQS_QUEUE_URL,
                MaxNumberOfMessages=1,
                WaitTimeSeconds=20,
            )
            for msg in resp.get('Messages', []):
                try:
                    self.process_message(msg)
                except Exception:
                    logger.exception("Failed to process job — leaving message in queue for retry")


if __name__ == '__main__':
    InterpolationWorker().run()
