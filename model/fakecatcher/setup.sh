#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Step 1: Install Miniconda (if not already installed)
if [ ! -d "$HOME/miniconda3" ]; then
    echo "Installing Miniconda..."
    mkdir -p ~/miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
    chmod +x ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    ~/miniconda3/bin/conda init bash
fi

# Reload shell for Conda initialization
echo "Reloading shell..."
source ~/miniconda3/etc/profile.d/conda.sh

# Step 2: Set up Conda environment
echo "Setting up Conda environment..."
conda create -n video python==3.11 -y || true
conda activate video

# Step 3: Install dependencies
echo "Installing dependencies..."
conda install conda-forge::libgl -y
conda install -c conda-forge ffmpeg -y
pip install -r requirements.txt
python -m pip install mediapipe

# step 4: Install zip files
apt-get update
apt-get install zip
cd /root/25th-conference-fakebusters/model/fakecatcher/data
mkdir manipulated_sequences
mkdir original_sequences
gdown 13cGlSlgv6-85hpEim0n9LwFHapK5vr9d
gdown 1SOfyJ_wdoFdroL--06FbQm3SDm_0Edsr
gdown 10Cfjfo_SwuOHXK-q-2B5d_j7tIDUcLPb
mv /manipulated_sequences_1/* /manipulated_sequences/
rm manipulated_sequences_1.zip
mv /manipulated_sequences_2/* /manipulated_sequences/
rm manipulated_sequences_2.zip
mv /actors /original_sequences/
mv /youtube /original_sequences/


echo "Setup completed successfully!"