# FakeCatcher Inference Code

This code was written by adapting the contents of paper [FakeCatcher: Detection of Synthetic Portrait Videos using Biological Signals] https://arxiv.org/abs/1901.02212 

## How to Use (CNN based Model)

### 1. Train step

#### 1-1. Preprocess PPG-map
```bash
cd /25th-conference-fakebusters/model/fakecatcher
python model/fakecatcher/cnn/preprocess_map.py -c model/fakecatcher/utils/config.yaml -l model/fakecatcher/data/ppg_map.log -o model/fakecatcher/data
```

#### 1-2. Train cnn model
```bash
cd /25th-conference-fakebusters/model/fakecatcher
python model/fakecatcher/cnn/train_cnn.py -c model/fakecatcher/utils/config.yaml -l model/fakecatcher/data/ppg_cnn.log -i model/fakecatcher/data/ppg_maps.json -o model/fakecatcher/model_state_cnn.pt
```

### 2. Inference step
```bash
cd /25th-conference-fakebusters/model/fakecatcher/cnn
python main.py -c model/fakecatcher/utils/config.yaml
```
