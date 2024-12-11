from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import boto3
import uuid
import os
import time
import csv
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://localhost:8080",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment variables
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
BUCKET_NAME = "fakebuster"
REGION = "ap-northeast-2"

# Initialize S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=REGION
)

CSV_FILE_PATH = "upload_metrics.csv"

# Ensure CSV file exists and has the correct header
if not os.path.exists(CSV_FILE_PATH):
    with open(CSV_FILE_PATH, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["file_name", "file_type", "file_size(B)", "upload_time(ms)"])

def log_upload_metrics(file_name, file_type, file_size, upload_time):
    with open(CSV_FILE_PATH, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([file_name, file_type, file_size, upload_time])




@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/api/files/upload")
async def upload_file(file: UploadFile = File(...)):
    start_time = time.time()
    print(f"Upload started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(start_time))}")

    file_key = f"{uuid.uuid4()}.{file.filename.split('.')[-1]}"
    s3_client.upload_fileobj(file.file, BUCKET_NAME, file_key)

    end_time = time.time()
    print(f"Upload finished at: {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(end_time))}")
    print(f"Upload duration: {int((end_time - start_time) * 1000)} ms")

    # Log metrics to CSV
    file_name = file.filename
    file_type = file.content_type
    contents = await file.read()
    file_size = len(contents)
    file.file.seek(0)
    upload_time = int((end_time - start_time) * 1000)

    log_upload_metrics(file_name, file_type, file_size, upload_time)

    return JSONResponse(content={"fileKey": file_key})
