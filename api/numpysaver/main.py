from fastapi import FastAPI, Request
import boto3
import numpy as np
import io
import os
import tempfile
app = FastAPI()
    

@app.post("/{hand_sign}/{object_name}")
async def save(hand_sign: str, object_name: str,request: Request):
    s3 =  boto3.client(
                        "s3",
                        endpoint_url=os.environ.get("S3_URL"),
                        aws_access_key_id=os.environ.get("S3_ACCESS_KEY"),
                        aws_secret_access_key=os.environ.get("S3_SECRET_KEY"),
                    )
    binary_data = await request.body()
    binary_file = io.BytesIO(binary_data)
    array = np.load(binary_file)

    print(array)

    with tempfile.NamedTemporaryFile() as temp:
        np.save(temp, array)
        temp.flush()
        temp.seek(0)
        s3.upload_fileobj(temp, os.environ.get("S3_BUCKET_NAME"), f"{hand_sign}/{object_name}")

    # s3.upload_fileobj(binary_file, os.environ.get("S3_BUCKET_NAME"), f"{hand_sign}/{object_name}")
    return {"message": "Filed has been saved"}