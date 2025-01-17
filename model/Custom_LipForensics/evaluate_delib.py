import argparse
import dlib
import yaml
import numpy as np
import ffmpeg
import os
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score

import sys
sys.path.insert(0, "./LipForensics")
from inference import evaluate_lipforensics
sys.path.insert(0, "/root/roi_extractor/utils") #여기 수정
from preprocess import VideoROIExtractor


def process_video(video_path, config):
    """비디오를 처리하여 cropped mouth 배열을 반환"""
    try:
        # dlib 모델 초기화
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(config['face_predictor_path'])
        mean_face = np.load(config['mean_face_path'])

        # VideoROIExtractor 초기화
        processor = VideoROIExtractor(
            input_video_path=video_path,
            detector=detector,
            predictor=predictor,
            mean_face=mean_face,
            config=config,
        )

        # 비디오 처리
        cropped_mouth_array = processor.preprocess_video()
        if cropped_mouth_array is None:
            print(f"Failed to process video: {video_path}")
            return None
        return cropped_mouth_array
    except Exception as e:
        print(f"Error processing video {video_path}: {e}")
        return None

def main():
    # argparse 설정
    parser = argparse.ArgumentParser(description="Process multiple videos from CSV and calculate accuracy.")
    parser.add_argument("--csv_file", type=str, default='/test_video_list_400.csv', help="Path to the CSV file with video paths and labels.")
    args = parser.parse_args()

    # Config 파일 읽기
    # 여기 수정
    with open('/root/roi_extractor/config.yaml', 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    # CSV 파일 읽기
    data = pd.read_csv(args.csv_file)

    # # 비디오 처리 및 레이블 예측
    # probabilities = []
    # true_labels = []

    for idx, row in tqdm(data.iterrows(), total=len(data), desc="Processing videos"):
        video_path = row['video_path']
        label = row['label']  # 1: Real, 0: Fake

        # 이미 처리된 확률값이 있다면 건너뜀
        if pd.notna(row['probability']):
            print('이미 처리된 확률값이 있음')
            continue

        # 비디오 존재 확인
        if not os.path.exists(video_path):
            print(f"Video not found: {video_path}")
            data.at[idx, 'probability'] = None
            data.to_csv(args.csv_file, index=False)
            continue

        # 비디오 처리
        cropped_mouth_array = process_video(video_path, config)
        if cropped_mouth_array is None:
            data.at[idx, 'probability'] = None
            data.to_csv(args.csv_file, index=False)
            continue

        # LipForensics inference 실행
        probability = evaluate_lipforensics(cropped_mouths_array=cropped_mouth_array)
        if probability < 0:
            data.at[idx, 'probability'] = None
            data.to_csv(args.csv_file, index=False)
            continue

        # 결과 반전: LipForensics 결과는 0=Real, 1=Fake이므로, 레이블과 반대
        probability = 1 - probability

        # 확률값 저장 및 CSV 파일 업데이트
        data.at[idx, 'probability'] = probability
        print(f"Processed {video_path}: Probability={probability:.4f}, Label={label}")
        data.to_csv(args.csv_file, index=False)

    # probability가 None인 행 제거한 복사본 생성
    valid_data = data.dropna(subset=['probability'])

    # Accuracy 계산
    true_labels = valid_data['label'].tolist()
    predicted_labels = [1 if p >= 0.5 else 0 for p in valid_data['probability']]
    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
