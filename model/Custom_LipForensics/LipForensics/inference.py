"""Evaluate pre-trained LipForensics model on various face forgery datasets"""

import argparse
from collections import defaultdict

import yaml
# import pandas as pd
# from sklearn import metrics
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Resize, Lambda
from tqdm import tqdm

import sys
sys.path.insert(0, "./LipForensics/data")
from transforms import NormalizeVideo, ToTensorVideo
from dataset_clips import SingleVideoClips
sys.path.insert(0, "./LipForensics/models")
from spatiotemporal_net import get_model

def evaluate_lipforensics(
    cropped_mouths_array,
    weights_forgery_path="./LipForensics/models/weights/lipforensics_ff.pth",
    frames_per_clip=25,
    batch_size=8,
    device="cuda:0",
    grayscale=True, #수정
    num_workers=4,
):
    """
    Evaluate LipForensics model on a single video.

    :param video_path: Path to a single video
    :param weights_forgery_path: Path to pretrained weights for forgery detection
    :param frames_per_clip: Number of frames per clip
    :param batch_size: Batch size for inference
    :param device: Device to put tensors on
    :param grayscale: Use grayscale frames if True, otherwise RGB
    :param num_workers: Number of workers for data loading
    :return: Prediction score (0: Real, 1: Fake)
    """
    # Load the pre-trained model
    model = get_model(weights_forgery_path=weights_forgery_path)

    # Transformation pipeline
    transform = Compose(
    [
        ToTensorVideo(),
        #Lambda(lambda x: torch.nn.functional.pad(x, (0, max(0, 88 - x.shape[2]), 0, max(0, 88 - x.shape[1])))),
        CenterCrop((88, 88)),
        NormalizeVideo((0.421,), (0.165,)),
    ])
        
    # Single video dataset
    dataset = SingleVideoClips(
        video_frames=cropped_mouths_array, #여기를 numpy array로 짠다.
        frames_per_clip=frames_per_clip,
        grayscale=grayscale,
        transform=transform,
    )
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, drop_last=False)

    # Inference
    model.eval()
    logits_list = []
    with torch.no_grad():
        for clips, _ in tqdm(loader, desc="Running LipForensics Inference"):
            clips = clips.to(device)
            logits = model(clips, lengths=[frames_per_clip] * clips.shape[0])
            logits_list.append(logits)
    
    #print(f"Number of clips in dataset: {len(logits_list)}")


    # Aggregate logits and compute prediction
    if len(logits_list) == 0:
        return -1
    avg_logits = torch.mean(torch.cat(logits_list), dim=0)
    prediction = torch.sigmoid(avg_logits).item()
    return prediction

if __name__ == "__main__":
    prediction = evaluate_lipforensics(video_path='../chimchak.mp4')
    print(f"Final Prediction: {prediction:.4f} (0: Real, 1: Fake)")