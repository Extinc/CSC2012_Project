from fastapi import FastAPI, Request, Response
import boto3
import numpy as np
import io
import os
import tempfile
app = FastAPI()
    

@app.get("/")
async def get():
    s3 =  boto3.client(
                        "s3",
                        endpoint_url=os.environ.get("S3_URL"),
                        aws_access_key_id=os.environ.get("S3_ACCESS_KEY"),
                        aws_secret_access_key=os.environ.get("S3_SECRET_KEY"),
                    )
    bucket_name = os.environ.get('S3_Bucket_Name')

    response = s3.get_object(Bucket=bucket_name, Key='hand_sign_model.h5')
    model_body = response['Body'].read()

    return Response(content=model_body, 
                    media_type='application/octet-stream' , 
                    headers={'Content-Disposition': 'attachment;filename=hand_sign_model.h5'})