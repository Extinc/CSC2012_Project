import hashlib
import boto3 
import mimetypes
import io
import pickle


import numpy as np

s3 =  boto3.client(
                "s3",
                endpoint_url="http://localhost:9000",
                    aws_access_key_id="minioaccesskey",
                    aws_secret_access_key="miniosecretkey",
                region_name="us-east-1",)

# Create a bucket
bucket_name = "state"
# response = s3.create_bucket(Bucket=bucket_name)

print(f"Bucket {bucket_name} created.")
response = s3.list_buckets()


# print("List of buckets:")
# for bucket in response["Buckets"]:
#     print(bucket["Name"])

s3r = boto3.resource('s3',
                     endpoint_url="http://localhost:9000",
                     aws_access_key_id="minioaccesskey",
                     aws_secret_access_key="miniosecretkey")

bucket = s3r.Bucket('data')
# with open('/Users/voidky/Documents/GitHub/CSC2012_Project/ky_1678585074.npy', 'rb') as data:
#     s3.upload_fileobj(data, bucket_name, 'ky_1678585074.npy')

# nump = np.load('/Users/voidky/Documents/GitHub/CSC2012_Project/ky_1678585074.npy')

# my_array_data = io.BytesIO()
# pickle.dump(nump, my_array_data)
# my_array_data.seek(0)
# s3.upload_fileobj(my_array_data, bucket_name, 'your-file.pkl')


response = s3.get_object(Bucket='data', Key='a/jf_1678778296_label.npy')
body = response['Body'].read()

my_array_data2 = io.BytesIO(body)

my_array2 = np.load(io.BytesIO(body))


print("RESULT: ",my_array2)

# print("BUCKET ITEM : ", np.load("/Users/voidky/Documents/GitHub/CSC2012_Project/ky_1678585074.npy"))