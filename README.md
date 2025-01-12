
# 페이크 버스터즈

<div align="center">
<h3>24-2 YBIGTA 컨퍼런스</h3>
<em>딥페이크 탐지 모델</em>
</div>

## 목차
- [문제 정의](#문제-정의)
- [선행 연구](#선행-연구)
- [세부 목표](#세부-목표)
- [접근 방법](#접근-방법)
- [모델별 접근 방법 및 특징](#모델별-접근-방법-및-특징)
- [결과 및 주요 기능](#결과-및-주요-기능)
- [팀 구성](#팀-구성)

## 문제 정의
### 불법 음란물 유포로 인한 피해 확산
생성형 AI의 발전과 함께 다양한 문제가 발생하고 있으며, 최근 주목받고 있는 문제 중 하나는 딥페이크 기술을 이용한 음란물 생성 및 유포입니다. 2023년 미국 사이버보안 업체 Security Hero의 보고서에 따르면, 딥페이크 포르노 범죄율이 가장 높은 나라는 대한민국으로, 사태의 심각성을 보여줍니다.

### 유명인사 이외의 일반인에 대한 딥페이크 영상 유포
기술 발전과 편리함의 증가로 인해 이제는 유명인뿐만 아니라 일반인들도 딥페이크 음란물의 피해자가 되고 있습니다. 예를 들어, 2023년 기준 딥페이크 음란물로 피해를 입은 10대 청소년은 86명이었으나, 2024년 8월까지 그 수가 238명으로 급증했습니다.

![stats](static/statsbycountry.png)

## 세부 목표
1. 범용적이고 광범위한 딥페이크 탐지 솔루션 개발
2. 각종 모델들의 성능 비교 및 분석
3. 실시간 딥페이크 탐지 서비스 구현

## 접근 방법
1. **태스크**
    - 데이터 수집 및 전처리
    - 모델별 탐지 알고리즘 구현 및 성능 평가
2. **데이터셋**
    - FaceForensics++, Celeb-DF-v2 등 딥페이크 관련 공개 데이터셋 사용
3. **모델링/아키텍처**
    - FakeCatcher, LipForensics, MMDet 모델 구현
    - 백엔드 및 프론트엔드 개발을 통한 사용자 인터페이스 구축

## 모델별 접근 방법 및 특징
| 모델명       | 주요 탐지 방식                    | 사용된 기술/알고리즘             | 주요 특징                            |
|--------------|----------------------------------|---------------------------------|-------------------------------------|
| FakeCatcher  | PPG(Photoplethysmography) 신호   | Dlib, MediaPipe, CNN, Random Forest | 생체 신호 기반 탐지, 얼굴 영역 중심 분석 |
| LipForensics | 입술 움직임 패턴 탐지             | ResNet-18, MSTCN                | 시공간적 움직임 분석, 다양한 위조 방식에 강건 |
| MMDet        | 이미지 주파수 도메인 분석         | CLIP Encoder, DIRE, VQ-VAE      | GAN 및 Diffusion 이미지 생성 방식 탐지 |

### 모델별 주요 기능 및 역할

#### 1. FakeCatcher
- **탐지 방식**: 얼굴 중심부의 PPG 신호를 분석하여 위조된 영상을 탐지합니다.
- **기술적 특징**: CPPG와 GPPG 신호를 통해 생체 신호의 변화 패턴을 포착하고, 이를 CNN과 Random Forest로 분류합니다.
- **장점**: 얼굴 영역에서 나타나는 미세한 생체 신호를 통해 기존 탐지 방법으로 놓칠 수 있는 위조 영상을 탐지할 수 있습니다.
- **한계**: 데이터 전처리 과정이 복잡하고, 실시간 탐지에는 시간이 많이 소요됩니다.

#### 2. LipForensics
- **탐지 방식**: 입술 움직임에서 나타나는 고차원적 의미적 불규칙성을 분석합니다.
- **기술적 특징**: ResNet-18 기반의 Feature Extractor와 MSTCN(Multi-Scale Temporal Convolutional Network)을 사용합니다.
- **장점**: 다양한 위조 방식에 대해 일반화된 탐지 성능을 보여주며, 데이터 변형에 강건합니다.
- **한계**: ROI 추출 방식에 따라 성능 차이가 발생할 수 있습니다.

#### 3. MMDet
- **탐지 방식**: GAN 및 Diffusion 모델로 생성된 이미지의 주파수 도메인에서 나타나는 시각적 아티팩트를 분석합니다.
- **기술적 특징**: CLIP Encoder, DIRE, VQ-VAE 등을 사용하여 프레임 단위로 영상을 분석합니다.
- **장점**: 얼굴 영역을 포착하지 못하는 경우에도 이미지 자체의 특성을 활용하여 딥페이크를 탐지할 수 있습니다.
- **한계**: Diffusion 모델 기반 탐지는 시간이 많이 소요되며, 실시간 응용에는 부적합할 수 있습니다.

## 결과 및 주요 기능
- 영상 데이터의 실시간 분석 및 탐지
- 사용자 인터페이스 제공을 위한 프론트엔드 개발
- 백엔드 서버를 통한 병렬 처리 및 최적화

## 팀 구성
|이름|팀|역할|
|-|-|-|
<<<<<<< Updated upstream
|박동연|DS|(역할)|
|이동렬|DS|(역할)|
|정회수|DA|(역할)|
|양인혜|DS|(역할)|
|임채림|DE|(역할)|
|성현준|DE|(역할)|
|정다연|DS|(역할)|
=======
|박동연|DS|모델링 및 데이터 처리|
|이동렬|DS|모델 성능 평가|
|정회수|DA|데이터 분석|
|양인혜|DS|데이터 전처리|
|임채림|DE|프론트엔드 개발|
|성현준|DE|백엔드 개발|
|정다연|DS|프로젝트 관리 및 리포트 작성|
>>>>>>> Stashed changes
