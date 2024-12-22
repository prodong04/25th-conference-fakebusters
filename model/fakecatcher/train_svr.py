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
from utils.logging import setup_logging

def majority_voting(probabilities):
    """
    Perform majority voting on the predicted probabilities.
    """
    mean_prob = np.mean(probabilities)
    majority_vote = np.round(mean_prob)
    return majority_vote

def extract_feature(video_path, true_label, config):
    """Process a single video to extract features."""
    landmarker = ROIProcessor(video_path, config)
    R_means_array, L_means_array, M_means_array, original_fps = landmarker.detect_with_calculate()

    if R_means_array.shape[0] == 0:
        logger.warning(f"Skipping video {video_path} because R_means_array is empty.")
        return None, None

    features = []
    time_interval = config['seg_time_interval']
    target_fps = config['fps_standard']

    for i in range(R_means_array.shape[0]):
        G_R = PPG_G.from_RGB(R_means_array[i], original_fps).compute_signal()
        G_L = PPG_G.from_RGB(L_means_array[i], original_fps).compute_signal()
        G_M = PPG_G.from_RGB(M_means_array[i], original_fps).compute_signal()
        C_R = PPG_C.from_RGB(R_means_array[i], original_fps).compute_signal()
        C_L = PPG_C.from_RGB(L_means_array[i], original_fps).compute_signal()
        C_M = PPG_C.from_RGB(M_means_array[i], original_fps).compute_signal()

        # Segment signals
        R_ROI_G_segments = frequency_resample(G_R, time_interval, original_fps, target_fps)
        R_ROI_C_segments = frequency_resample(C_R, time_interval, original_fps, target_fps)
        L_ROI_G_segments = frequency_resample(G_L, time_interval, original_fps, target_fps)
        L_ROI_C_segments = frequency_resample(C_L, time_interval, original_fps, target_fps)
        M_ROI_G_segments = frequency_resample(G_M, time_interval, original_fps, target_fps)
        M_ROI_C_segments = frequency_resample(C_M, time_interval, original_fps, target_fps)

        # Combine segments
        ppg = [
            L_ROI_G_segments, M_ROI_G_segments, R_ROI_G_segments,
            L_ROI_C_segments, M_ROI_C_segments, R_ROI_C_segments
        ]

        # Extract features
        fe = FeatureExtractor(target_fps, *ppg)
        features.append(fe.feature_union())

    return np.array(features), true_label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_path', type=str, required=True, help="Path to the config file.")
    args = parser.parse_args()

    # Load configuration
    with open(args.config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    # Set up logging
    global logger
    logger = setup_logging()

    # Load data
    data_root_directory = config['data_root_directory']
    video_paths, true_labels = load_data(data_root_directory, config["meta_data_csv_path"])
    logger.info(f"Loaded {len(video_paths)} videos.")

    # Extract features
    video_features, video_labels = [], []
    for video_path, true_label in tqdm(zip(video_paths, true_labels), desc='Processing videos'):
        features, label = extract_feature(video_path, true_label, config)
        if features is not None:
            video_features.append(features)
            video_labels.append(label)

    # Combine all features and labels
    all_features = np.vstack(video_features)
    all_labels = np.hstack([[label] * len(features) for label, features in zip(video_labels, video_features)])

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(all_features, all_labels, test_size=0.3, random_state=42)

    # Train SVR model
    model = SVRModel()
    model.train(X_train, y_train)
    model.save_model(config.get('trained_model_path', 'svr_model.pkl'))
    logger.info("Model trained and saved.")

    # Evaluate model
    correct_predictions = 0
    for features, actual_label in zip(video_features, video_labels):
        segment_probabilities = model.predict(features)
        predicted_label = majority_voting(segment_probabilities)
        logger.info(f"Actual: {actual_label}, Predicted: {predicted_label}")

        if predicted_label == actual_label:
            correct_predictions += 1

    accuracy = correct_predictions / len(video_features)
    logger.info(f"Video-level Accuracy: {accuracy * 100:.2f}%")
    print(f"Video-level Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
