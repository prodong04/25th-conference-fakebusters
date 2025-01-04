import argparse
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score

import sys
sys.path.insert(0, "/root/25th-conference-fakebusters/model/Custom_LipForensics/LipForensics")
from inference import evaluate_lipforensics

sys.path.insert(0, "/root/25th-conference-fakebusters/model/Custom_LipForensics/utils")
from mouth_roi import ROIProcessor
import os


def process_video(video_path, config):
    """
    비디오를 처리하여 cropped mouth 배열을 반환
    """
    try:
        # ROIProcessor 초기화
        landmarker = ROIProcessor(video_path=video_path, config=config)
        # 입 영역 크롭
        cropped_mouths_array, _ = landmarker.detect_with_crop()

        # numpy 배열로 변환
        cropped_mouths_array = np.array(cropped_mouths_array)

        if len(cropped_mouths_array) == 0 or cropped_mouths_array is None:
            print(f"No cropped mouths detected in video: {video_path}")
            return None
        return cropped_mouths_array
    except Exception as e:
        print(f"Error processing video {video_path}: {e}")
        return None


def main():
    # argparse 설정
    parser = argparse.ArgumentParser(description="Evaluate multiple videos from CSV and calculate accuracy.")
    parser.add_argument("--csv_file", type=str, required=True, help="Path to the CSV file with video paths and labels.")
    parser.add_argument("--start_row", type=int, default=0, help="Row index to start processing (0-based index).")
    args = parser.parse_args()

    # Config 파일 읽기
    config_path = '/root/25th-conference-fakebusters/model/Custom_LipForensics/utils/config.yaml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # CSV 파일 읽기
    data = pd.read_csv(args.csv_file)

    '''# 시작 행 지정
    if args.start_row > 0:
        
        print(f"Processing starts from row {args.start_row}.")
        data = data.iloc[args.start_row:].reset_index(drop=True)
'''
    # probability 열 초기화
    if 'probability' not in data.columns:
        data['probability'] = None

    # 비디오 처리 및 레이블 예측
    for idx, row in tqdm(data.iterrows(), total=len(data), desc="Processing videos"):
        '''if idx < args.start_row:
            continue'''
        video_path = row['video_path']
        label = row['label']  # 1: Real, 0: Fake

        # 이미 처리된 확률값이 있다면 건너뜀
        if pd.notna(row['probability']):
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
        try:
            probability = evaluate_lipforensics(cropped_mouths_array=cropped_mouth_array)
            if probability < 0:
                data.at[idx, 'probability'] = None
                data.to_csv(args.csv_file, index=False)
                continue
            # 결과 반전: LipForensics 결과는 0=Real, 1=Fake이므로, 레이블과 반대
            probability = 1 - probability
        except Exception as e:
            print(f"Error during inference for video {video_path}: {e}")
            data.at[idx, 'probability'] = None
            data.to_csv(args.csv_file, index=False)
            continue

        # 확률값 저장 및 CSV 파일 업데이트
        data.at[idx, 'probability'] = probability
        print(f"Processed {video_path}: Probability={probability:.4f}, Label={label}")
        data.to_csv(args.csv_file, index=False)

        
    print(f"Final updated CSV saved to {args.csv_file}")

    # probability가 None인 행 제거한 복사본 생성
    valid_data = data.dropna(subset=['probability'])

    # Accuracy 계산
    true_labels = valid_data['label'].tolist()
    predicted_labels = [1 if p >= 0.5 else 0 for p in valid_data['probability']]
    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"Accuracy: {accuracy:.4f}")



if __name__ == "__main__":
    main()
