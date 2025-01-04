import argparse
import yaml
import numpy as np
import ffmpeg

import sys
sys.path.insert(0, "/root/25th-conference-fakebusters/model/Custom_LipForensics/LipForensics")
from inference import evaluate_lipforensics

sys.path.insert(0, "/root/25th-conference-fakebusters/model/Custom_LipForensics/utils")
from mouth_roi import ROIProcessor
import cv2
import os

def main():
    # argparse 설정
    parser = argparse.ArgumentParser(description="Extract frames from a video and process with VideoROIExtractor.")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the input video file.")
    args = parser.parse_args()

    config_path = '/root/25th-conference-fakebusters/model/Custom_LipForensics/utils/config.yaml'

    # Config 파일 읽기
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    landmarker = ROIProcessor(video_path=args.video_path, config=config)

    #cropped_mouths_arrary는 list
    cropped_mouths_array, fps = landmarker.detect_with_crop()
    

    #cropped_mouths_array_draw, fps_draw = landmarker.detect_with_draw()
    #cropped_mouth_video = landmarker.save_cropped_video(cropped_mouths_array_draw, fps_draw)
    #print("cropped_mouth_video_path:", cropped_mouth_video)

    cropped_mouths_array = np.array(cropped_mouths_array)

    '''# 4.5 동영상 저장
    # 동영상 저장 디렉토리
    output_dir = "/root/25th-conference-fakebusters/model/Custom_LipForensics/cropped_mouth_video"
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
            s=f"{cropped_mouths_array.shape[2]}x{cropped_mouths_array.shape[1]}"  # 너비x높이
        )
        .output(output_path, pix_fmt='yuv420p', vcodec='libx264', r=25)  # 출력 설정 (H.264 코덱, 25 FPS)
        .overwrite_output()
        .run_async(pipe_stdin=True)  # 파이프 입력 비동기로 실행
    )

    # 프레임 데이터를 ffmpeg 프로세스로 전달
    for frame in cropped_mouths_array:
        process.stdin.write(frame.tobytes())

    # ffmpeg 프로세스 종료
    process.stdin.close()
    process.wait()

    print(f"output_path: {output_path}")'''

    # LipForensics
    prediction = evaluate_lipforensics(cropped_mouths_array=cropped_mouths_array)
    print(f"Final Prediction: {prediction:.4f}")

if __name__ == "__main__":
    main()
