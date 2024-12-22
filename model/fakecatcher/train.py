import yaml
from tqdm import tqdm
import logging
import argparse
import numpy as np
from sklearn.model_selection import train_test_split

from utils.roi import ROIProcessor
from fakeavceleb import load_data
from utils.ppg.ppg_c import PPG_C
from utils.ppg.ppg_g import PPG_G
from utils.feature.feature_extractor import FeatureExtractor
from svr_model import SVRModel


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
    features_list = []
    video_labels = []  # 비디오 단위 레이블 저장

    for video_path, true_label in tqdm(zip(video_paths, true_labels), desc='training'):
        try:
            
            landmarker = ROIProcessor(video_path, config["model_path"])
            R_means_dict, L_means_dict, M_means_dict, fps = landmarker.detect_with_calculate()
            
            # Compute PPG signals
            G_R, G_L, G_M = [PPG_G(d, fps).compute_signal() for d in [R_means_dict, L_means_dict, M_means_dict]]
            C_R, C_L, C_M = [PPG_C(d, fps).compute_signal() for d in [R_means_dict, L_means_dict, M_means_dict]]
            
            # Segment signals
            R_ROI_G_segments, R_ROI_C_segments = split_segments(G_R, 50), split_segments(C_R, 50)
            L_ROI_G_segments, L_ROI_C_segments = split_segments(G_L, 50), split_segments(C_L, 50)
            M_ROI_G_segments, M_ROI_C_segments = split_segments(G_M, 50), split_segments(C_M, 50)

            # Combine segments
            combined_segments = combine_segments(
                R_ROI_G_segments, R_ROI_C_segments, 
                L_ROI_G_segments, L_ROI_C_segments, 
                M_ROI_G_segments, M_ROI_C_segments
            )

            # Extract features for each segment
            features = []
            for ppg in combined_segments:
                fe = FeatureExtractor(fps, *ppg)
                features.append(fe.feature_union())
            
            features_list.append(np.array(features))
            video_labels.append(true_label)

        except Exception as e:
            logger.error(f"Error processing video {video_path}: {e}", exc_info=True)
            break

    # Combine all features and labels for training
    all_features = np.vstack(features_list)
    all_labels = np.hstack([[label] * len(features) for label, features in zip(video_labels, features_list)])
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(all_features, all_labels, test_size=0.3, random_state=42)

    # Train SVR Model
    model = SVRModel()
    model.train(X_train, y_train)

    # Predict video labels using majority voting
    correct_predictions = 0
    total_videos = len(features_list)

    for features, actual_label in zip(features_list, video_labels):
        segment_probabilities = model.predict(features)  # 각 세그먼트 확률 예측
        predicted_label = majority_voting(segment_probabilities)  # 다수결 투표로 비디오 레이블 결정
        print(f"Actual: {actual_label}, Predicted: {predicted_label}")

        if predicted_label == actual_label:
            correct_predictions += 1

    # 전체 정확도 계산
    accuracy = correct_predictions / total_videos
    print(f"Video-level Accuracy: {accuracy * 100:.2f}%")

    # 모델 저장
    model.save_model('svr_model.pkl')