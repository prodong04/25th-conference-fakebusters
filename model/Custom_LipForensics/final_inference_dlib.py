##################
#서버랑 연결되어 있음
##################

import argparse
import dlib
import yaml
import numpy as np
import ffmpeg
import numpy as np

import sys
sys.path.insert(0, "./LipForensics")
from inference import evaluate_lipforensics
sys.path.insert(0, "/root/roi_extractor/utils") #여기 수정
from preprocess import VideoROIExtractor
import os
import cv2

def main():
    # argparse 설정
    parser = argparse.ArgumentParser(description="Extract frames from a video and process with VideoROIExtractor.")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the input video file.")
    #parser.add_argument("--config_path", type=str, required=True, help="Path to the configuration YAML file.")
    args = parser.parse_args()

    # Config 파일 읽기
    # 여기 수정
    with open('/root/roi_extractor/config.yaml', 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    # dlib 모델 초기화
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(config['face_predictor_path'])
    mean_face = np.load(config['mean_face_path'])

    # VideoROIExtractor 초기화
    processor = VideoROIExtractor(
        input_video_path=args.video_path,
        detector=detector,
        predictor=predictor,
        mean_face=mean_face,
        config=config
    )

    # 1~4. 비디오 처리 (프레임 추출 및 ROI 크롭)
    cropped_mouth_array = processor.preprocess_video() # 비디오에서 크롭된 입 영역 추출
    if cropped_mouth_array is None:
        print("Failed to process video with VideoROIExtractor.")
        return
    print("Cropped mouths successfully extracted.")

    # 4.5 동영상 저장
    # 동영상 저장 디렉토리
    output_dir = "./cropped_mouth_video"
    os.makedirs(output_dir, exist_ok=True)  # 디렉토리가 없으면 생성

    # 비디오 이름 생성
    video_name = os.path.basename(args.video_path).replace(".mp4", "_cropped.mp4")
    output_path = os.path.join(output_dir, video_name)

    # ffmpeg 입력 스트림 생성
    process = (
        ffmpeg
        .input(
            'pipe:',  # 데이터를 파이프로 전달
            format='rawvideo',
            pix_fmt='rgb24',
            s=f"{cropped_mouth_array.shape[2]}x{cropped_mouth_array.shape[1]}"  # 너비x높이
        )
        .output(output_path, pix_fmt='yuv420p', vcodec='libx264', r=25)  # 출력 설정 (H.264 코덱, 25 FPS)
        .overwrite_output()
        .run_async(pipe_stdin=True)  # 파이프 입력 비동기로 실행
    )

    # 프레임 데이터를 ffmpeg 프로세스로 전달
    for frame in cropped_mouth_array:
        process.stdin.write(frame.tobytes())

    # ffmpeg 프로세스 종료
    process.stdin.close()
    process.wait()

    print(f"output_path: {output_path}")
    
    
    # 5. LipForensics
    prediction = evaluate_lipforensics(cropped_mouths_array=cropped_mouth_array)
    print(f"Final Prediction: {prediction:.4f}")


if __name__ == "__main__":
    main()
