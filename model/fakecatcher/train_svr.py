import yaml
from tqdm import tqdm
import logging
import argparse
import numpy as np
from sklearn.model_selection import train_test_split

from utils.roi import ROIProcessor
from data.fakeavceleb import load_data
from utils.ppg.ppg_c import PPG_C
from utils.ppg.ppg_g import PPG_G
from utils.ppg.interpolate import frequency_resample
from utils.feature.feature_extractor import FeatureExtractor
from svr.svr_model import SVRModel


def split_segments(data, segment_length):
    """
    데이터를 segment_length 단위로 분할합니다. 
    남는 요소는 버립니다.
    
    Args:
        data (list or numpy.ndarray): 분할할 데이터.
        segment_length (int): 분할 크기.
        
    Returns:
        list: 분할된 세그먼트 리스트.
    """
    # 유효한 길이를 segment_length의 배수로 제한
    valid_length = len(data) - (len(data) % segment_length)
    return [np.array(data[i:i + segment_length]) for i in range(0, valid_length, segment_length)]

def combine_segments(*segments):
    """
    여러 세그먼트를 같은 인덱스 기준으로 묶습니다.

    Args:
        segments (list of lists): 여러 세그먼트 리스트.
    
    Returns:
        list of tuples: 인덱스별로 묶인 세그먼트 튜플 리스트.
    """
    return list(zip(*segments))

def majority_voting(probabilities):
    """
    Perform majority voting on the predicted probabilities.
    """
    mean_prob = np.mean(probabilities)
    majority_vote = np.round(mean_prob)
    return majority_vote

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_path', type=str, required=True, help="Path to the config file.")
    args = parser.parse_args()

    with open(args.config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    file_handler = logging.FileHandler('app.log', mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    root_directory = '/root/audio-visual-forensics/data'
    # Load Data
    video_paths, true_labels = load_data(root_directory, config["meta_data_csv_path"])
    
    # Feature Extraction
    video_features = []
    video_labels = []  # 비디오 단위 레이블 저장
    time_interval = config['seg_time_interval']
    target_fps = config['fps_standard']

    for video_path, true_label in tqdm(zip(video_paths, true_labels), desc='training'):
        try:
            
            landmarker = ROIProcessor(video_path, config)
            R_means_array, L_means_array, M_means_array, original_fps = landmarker.detect_with_calculate()
            
            if R_means_array.shape[0] == 0:
                logger.warning(f"Skipping video {video_path} because R_means_array is empty.")
                continue

            features = []
            for i in range(R_means_array.shape[0]):
                G_R = PPG_G.from_RGB(R_means_array[i], original_fps).compute_signal()
                G_L = PPG_G.from_RGB(L_means_array[i], original_fps).compute_signal() 
                G_M = PPG_G.from_RGB(M_means_array[i], original_fps).compute_signal() 
                C_R = PPG_C.from_RGB(R_means_array[i], original_fps).compute_signal()
                C_L = PPG_C.from_RGB(L_means_array[i], original_fps).compute_signal()
                C_M = PPG_C.from_RGB(M_means_array[i], original_fps).compute_signal()
            
                # Segment signals(1차원 np)
                R_ROI_G_segments, R_ROI_C_segments = frequency_resample(G_R, time_interval, original_fps, target_fps), frequency_resample(C_R, time_interval, original_fps, target_fps)
                L_ROI_G_segments, L_ROI_C_segments = frequency_resample(G_L, time_interval, original_fps, target_fps), frequency_resample(C_L, time_interval, original_fps, target_fps)
                M_ROI_G_segments, M_ROI_C_segments = frequency_resample(G_M, time_interval, original_fps, target_fps), frequency_resample(C_M, time_interval, original_fps, target_fps)

                # [G_L, G_M, G_R, C_L, C_M, C_R]이 한 행임임
                ppg = [
                    L_ROI_G_segments,
                    M_ROI_G_segments,
                    R_ROI_G_segments,
                    L_ROI_C_segments,
                    M_ROI_C_segments,
                    R_ROI_C_segments
                ]
           
                # Extract features for each segment
                fe = FeatureExtractor(config['fps_standard'], *ppg)

                features.append(fe.feature_union())
            
            video_features.append(np.array(features))
            video_labels.append(true_label)

        except Exception as e:
            logger.error(f"Error processing video {video_path}: {e}", exc_info=True)
            exit()

    # Combine all features and labels for training
    breakpoint()
    all_features = np.vstack(video_features)
    all_labels = np.hstack([[label] * len(features) for label, features in zip(video_labels, video_features)])
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(all_features, all_labels, test_size=0.3, random_state=42)

    # Train SVR Model
    model = SVRModel()
    model.train(X_train, y_train)

    model.save_model('svr_model.pkl')

    # Predict video labels using majority voting
    correct_predictions = 0
    total_videos = len(video_features)

    for features, actual_label in zip(video_features, video_labels):
        segment_probabilities = model.predict(features)  # 각 세그먼트 확률 예측
        predicted_label = majority_voting(segment_probabilities)  # 다수결 투표로 비디오 레이블 결정
        print(f"Actual: {actual_label}, Predicted: {predicted_label}")

        if predicted_label == actual_label:
            correct_predictions += 1

    # 전체 정확도 계산
    accuracy = correct_predictions / total_videos
    print(f"Video-level Accuracy: {accuracy * 100:.2f}%")

