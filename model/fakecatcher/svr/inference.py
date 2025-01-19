import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from pycaret.classification import load_model, predict_model
from preprocess_feature import extract_feature


def inference(video_path: str, config_path: str):
    # Load saved model and PCA
    model = load_model('../misc/svm_model')

    # Load configuration
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)


    # Extract features
    features = extract_feature(video_path, config)

    if not isinstance(features, np.ndarray):
        raise ValueError("Video에서 PPG를 찾을 수 없습니다. ")

    # Combine features and labels
    all_features = np.vstack(features)

    # Normalize features
    scaler = StandardScaler()
    all_features = scaler.fit_transform(all_features)
    
    df = pd.DataFrame(all_features, columns=[f"feature_{i}" for i in range(all_features.shape[1])])
    # Predict using the loaded model
    holdout_pred1 = predict_model(model, data=df)
    score = np.mean(holdout_pred1['prediction_label'])
    return score

if __name__=="__main__":
    video_path = './000.mp4'
    config = '/root/25th-conference-fakebusters/model/fakecatcher/utils/config.yaml'
    print(inference(video_path, config))