from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
import subprocess
import io
import tempfile
import cv2
import ast
import re
import os
import shutil
import numpy as np

app = FastAPI()

# 비디오 저장 디렉토리 설정
VIDEO_UPLOAD_DIR = "./uploaded_videos"
os.makedirs(VIDEO_UPLOAD_DIR, exist_ok=True)  # 디렉토리가 없으면 생성

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}


@app.post("/upload-video/")
async def upload_video(file: UploadFile = File(...)):
    """
    비디오 파일을 업로드받아 서버에 저장하지 않고 메모리에서 처리하는 API
    """
    # 비디오 파일 MIME 타입 확인
    if file.content_type not in ["video/mp4", "video/avi", "video/mov", "video/mkv"]:
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload a valid video file.")
    
    # 비디오 파일 저장 경로 설정
    video_path = os.path.join(VIDEO_UPLOAD_DIR, file.filename)

    # 비디오 파일 저장
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # final_inference.py 실행
    process = subprocess.run(
        ["python", "./final_inference_dlib.py", 
         "--video_path", video_path],
        capture_output=True,
        text=True
    )

    # print("STDOUT:", process.stdout)
    # print("STDERR:", process.stderr)

    
    if process.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Error running final_inference.py: {process.stderr}")

    # 결과 처리
    output = process.stdout
    cropped_video_path = None
    prediction_score = None

    # 결과 파싱
    for line in output.splitlines():
        if "Final Prediction" in line:
            prediction_score = float(line.split(":")[-1].strip().split()[0])
        elif "output_path" in line:
            cropped_video_path = line.split(":")[-1].strip()

    if prediction_score is None:
         raise HTTPException(status_code=500, detail="prediction_score is None")
    elif cropped_video_path is None:
         raise HTTPException(status_code=500, detail="cropped_video_path is None")
    

    cropped_video_path = open(cropped_video_path, "rb")
    response = StreamingResponse(cropped_video_path, media_type="video/mp4")
    response.headers["Score"] = f"{prediction_score}"
    return response