import os
import cv2
import dlib
import shutil
import skvideo
import tempfile
import skvideo.io
import subprocess
import numpy as np
from tqdm import tqdm
from collections import deque
from skimage import transform as tf


## ==================== LANDMARK ====================
def detect_landmark(image: np.ndarray,
                    detector: dlib.fhog_object_detector,
                    predictor: dlib.shape_predictor) -> np.ndarray:
    """
    이미지로부터 얼굴 랜드마크 추출.
    
    Args: 
        image: 비디오 프레임 이미지
        detector: 얼굴 검출용 오브젝트
        predictor: 랜드마크 검출용 오브젝트

    Returns:
        coords: 이미지 속 얼굴에 대한 랜드마크 좌표값
    """
    ## 이미지 길이, 너비 저장.
    img_height, img_width = image.shape[:2]
    ## 이미지 그레이스케일로 변환.
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    ## 얼굴이 포함된 바운딩 박스들을 모두 받아온다.
    rects = detector(gray, 1)
    ## 얼굴 랜드마크 리스트 초기화.
    coords = None

    for (_, rect) in enumerate(rects):

        ## 이미지 경계를 벗어나지 않도록 좌표 조정
        ## 참고로 (0,0)은 이미지 왼쪽 상단 모서리를 나타낸다.
        top = max(0, rect.top())
        bottom = min(rect.bottom(), img_height)
        left = max(0, rect.left())
        right = min(rect.right(), img_width)
        
        ## dlib.rectange 객체는 immutable.
        ## 유효한 바운딩 박스를 생성하여 얼굴 영역을 추출
        adjusted_rect = dlib.rectangle(left, top, right, bottom)

        ## 바운딩 박스 내부의 랜드마크 추출.
        shape = predictor(gray, adjusted_rect)
        ## 박스 별 랜드마크 좌표값 초기화.
        coords = np.zeros((68, 2), dtype=np.int32)

        for i in range(0, 68):
            ## 랜드마크 내부 포인트 별로 좌표값 할당.
            coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def linear_interpolate(landmarks: list,
                       start_idx: int,
                       stop_idx: int) -> list:
    """
    얼굴 랜드마크 추출을 실패한 프레임에 대한 선형 보간 작업 실행.
    
    Args: 
        landmarks: 비디오 하나에 대해 프레임 단위로 추출한 얼굴 랜드마크 리스트.
        start_idx: 결측치 발생 직전의 랜드마크
        stop_idx: 결측치 발생 이후 최초의 랜드마크

    Returns:
        landmarks: 선형 보간 작업을 거친 얼굴 랜드마크 리스트.
    """
    ## 결측치 발생 직전의 랜드마크와 결측치 발생 이후 최초의 랜드마크 불러오고, 값 차이 계산.
    start_landmarks = landmarks[start_idx]
    stop_landmarks = landmarks[stop_idx]
    delta = stop_landmarks - start_landmarks

    ## 선형 보간 작업.
    for idx in range(1, stop_idx-start_idx):
        landmarks[start_idx+idx] = start_landmarks + idx/float(stop_idx-start_idx) * delta
    return landmarks

def landmarks_interpolate(landmarks: list) -> list:
    """
    얼굴 랜드마크 추출을 실패한 프레임 구간 탐지 후 선형 보간.
    
    Args: 
        landmarks: 비디오 하나에 대해 프레임 단위로 추출한 얼굴 랜드마크 리스트.

    Returns:
        landmarks: 선형 보간 작업을 거친 얼굴 랜드마크 리스트.
    """
    ## 랜드마크가 제대로 추출된 프레임들의 인덱스만 저장하는 리스트.
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    
    ## 추출된 랜드마크가 하나도 없으면 None 반환.
    ## for-loop를 돌면서 연속적으로 잘 추출된 프레임들은 무시한다.
    ## 인덱스 간 차이가 1보다 크면 바로 linear_interpolate 메소드에 전달.
    if not valid_frames_idx:
        return None
    for idx in range(1, len(valid_frames_idx)):
        if valid_frames_idx[idx] - valid_frames_idx[idx-1] == 1:
            continue
        else:
            landmarks = linear_interpolate(landmarks, valid_frames_idx[idx-1], valid_frames_idx[idx])

    ## 마지막으로 비디오 시작과 끝 부분에 대한 랜드마크 결측 발생 시
    ## 가장 첫 랜드마크와 가장 마지막 랜드마크를 단순히 복사+붙여넣기 하는 방식으로 보간.
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    if valid_frames_idx:
        landmarks[:valid_frames_idx[0]] = [landmarks[valid_frames_idx[0]]] * valid_frames_idx[0]
        landmarks[valid_frames_idx[-1]:] = [landmarks[valid_frames_idx[-1]]] * (len(landmarks) - valid_frames_idx[-1])
    
    ## 최종 결측치 체크.
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    assert len(valid_frames_idx) == len(landmarks), "not every frame has landmark"
    return landmarks


## ====================   FACE   ====================
def warp_img(src: np.ndarray,
             dst: np.ndarray,
             img: np.ndarray,
             std_size: tuple[int, int]) -> tuple[np.ndarray, tf.SimilarityTransform]:
    """
    주어진 좌표들을 기반으로 이미지를 변환 후 변환 행렬과 함께 std_size로 지정된 크기로 출력.

    Args:
        src: 원본 이미지에서의 좌표 (N x 2 배열)
        dst: 목표 이미지에서의 좌표 (N x 2 배열)
        img: 변형할 이미지 (2D 또는 3D numpy 배열)
        std_size: 변형 후 출력될 이미지의 크기 (height, width)

    Returns:
        warped: 변형된 이미지 (uint8 타입)
        tform: 유사 변환 행렬
    """
    # 유사 변환 행렬 계산
    tform = tf.estimate_transform('similarity', src, dst)
    # 이미지 변형
    warped = tf.warp(img, inverse_map=tform.inverse, output_shape=std_size)
    # 0~1 범위에서 0~255 범위로 스케일링
    warped = warped * 255
    # uint8 형식으로 변환
    warped = warped.astype('uint8')
    return warped, tform

def apply_transform(transform: tf.SimilarityTransform,
                    img: np.ndarray,
                    std_size: tuple[int, int]) -> np.ndarray:
    """
    주어진 변환 행렬을 사용하여 이미지를 변환하고, 변환된 이미지를 반환.

    Args:
        transform: 유사 변환 행렬 (tf.SimilarityTransform)
        img: 변형할 이미지 (2D 또는 3D numpy 배열)
        std_size: 변형 후 출력될 이미지의 크기 (height, width)

    Returns:
        warped: 변형된 이미지 (uint8 타입)
    """
    # 변환 적용
    warped = tf.warp(img, inverse_map=transform.inverse, output_shape=std_size)
    # 0~1 범위에서 0~255 범위로 스케일링
    warped = warped * 255
    # uint8 형식으로 변환
    warped = warped.astype('uint8')
    return warped

def get_frame_count(filename: str) -> int:
    """
    비디오 파일의 총 프레임 수를 반환.

    Args:
        filename (str): 비디오 파일의 경로

    Returns:
        int: 비디오 파일의 총 프레임 수
    """
    cap = cv2.VideoCapture(filename)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total

def read_video(filename: str) -> np.ndarray:
    """
    비디오 파일을 프레임 단위로 읽어서 반환.

    Args:
        filename (str): 비디오 파일의 경로

    Yields:
        np.ndarray: 비디오의 각 프레임 (BGR 형식)
    """
    cap = cv2.VideoCapture(filename)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            yield frame
        else:
            break
    cap.release()


## ====================   CROP   ====================
def cut_patch(img: np.ndarray,
              landmarks: np.ndarray,
              height: int,
              width: int,
              threshold: int = 5) -> np.ndarray:
    """
    주어진 이미지에서 랜드마크를 기준으로 패치를 잘라서 반환.

    Args:
        img (np.ndarray): 입력 이미지 (2D 또는 3D numpy 배열)
        landmarks (np.ndarray): 랜드마크 좌표 (N x 2 배열)
        height (int): 잘라낼 패치의 높이
        width (int): 잘라낼 패치의 너비
        threshold (int, optional): 잘라낼 영역이 너무 크거나 작은 경우를 방지하는 임계값 (기본값은 5)

    Raises:
        Exception: 이미지의 크기와 잘라낼 패치 크기 차이가 너무 클 경우 예외를 발생시킴

    Returns:
        np.ndarray: 잘라낸 이미지 패치
    """
    # 랜드마크 중심 계산
    center_x, center_y = np.mean(landmarks, axis=0)

    # 이미지 크기를 벗어나지 않도록 중심 좌표 조정
    if center_y - height < 0:
        center_y = height
    if center_y - height < 0 - threshold:
        raise Exception('too much bias in height')
    if center_x - width < 0:
        center_x = width
    if center_x - width < 0 - threshold:
        raise Exception('too much bias in width')

    # 잘라낼 영역이 이미지 크기를 넘지 않도록 조정
    if center_y + height > img.shape[0]:
        center_y = img.shape[0] - height
    if center_y + height > img.shape[0] + threshold:
        raise Exception('too much bias in height')
    if center_x + width > img.shape[1]:
        center_x = img.shape[1] - width
    if center_x + width > img.shape[1] + threshold:
        raise Exception('too much bias in width')

    # 이미지에서 패치 추출
    cut_img = np.copy(
        img[int(round(center_y) - round(height)): int(round(center_y) + round(height)),
            int(round(center_x) - round(width)): int(round(center_x) + round(width))]
    )
    return cut_img

def crop_patch(video_pathname: str,
               landmarks: list[np.ndarray],
               mean_face_landmarks: np.ndarray,
               stablePntsIDs: list[int],
               STD_SIZE: tuple[int, int],
               window_margin: int,
               start_idx: int,
               stop_idx: int,
               crop_height: int,
               crop_width: int) -> np.ndarray:
    """
    주어진 비디오에서 입술 패치를 자르고, 그 패치들을 시퀀스로 반환.

    Args:
        video_pathname (str): 비디오 파일 경로
        landmarks (list of np.ndarray): 각 프레임에 대한 보간된 랜드마크 (각각 N x 2 배열)
        mean_face_landmarks (np.ndarray): 평균 얼굴 랜드마크 (N x 2 배열)
        stablePntsIDs (list of int): 안정적인 랜드마크 지점들의 인덱스 리스트
        STD_SIZE (tuple of int): 출력 이미지 크기 (height, width)
        window_margin (int): 비디오 프레임에서 사용할 윈도우 크기 (이전 및 이후 프레임 포함)
        start_idx (int): 입술 랜드마크 시작 인덱스
        stop_idx (int): 입술 랜드마크 끝 인덱스
        crop_height (int): 잘라낼 패치의 높이
        crop_width (int): 잘라낼 패치의 너비

    Returns:
        np.ndarray: 잘라낸 입술 패치 시퀀스 (형상: (프레임 수, crop_height, crop_width, 채널))
    
    Raises:
        Exception: 랜드마크를 처리할 수 없는 경우 예외 발생
    """
    frame_idx = 0
    num_frames = get_frame_count(video_pathname)
    frame_gen = read_video(video_pathname)
    margin = min(num_frames, window_margin)
    
    while True:
        try:
            frame = next(frame_gen)  # BGR 형식으로 프레임 읽기
        except StopIteration:
            break

        if frame_idx == 0:
            q_frame, q_landmarks = deque(), deque()
            sequence = []

        q_landmarks.append(landmarks[frame_idx])
        q_frame.append(frame)

        if len(q_frame) == margin:
            smoothed_landmarks = np.mean(q_landmarks, axis=0)  # 랜드마크 평균화 (스무딩)
            cur_landmarks = q_landmarks.popleft()
            cur_frame = q_frame.popleft()

            # 변환 적용
            trans_frame, trans = warp_img(smoothed_landmarks[stablePntsIDs, :],
                                          mean_face_landmarks[stablePntsIDs, :],
                                          cur_frame,
                                          STD_SIZE)
            trans_landmarks = trans(cur_landmarks)
            
            # 입술 패치 자르기
            sequence.append(cut_patch(trans_frame,
                                      trans_landmarks[start_idx:stop_idx],
                                      crop_height // 2,
                                      crop_width // 2))

        if frame_idx == len(landmarks) - 1:
            while q_frame:
                cur_frame = q_frame.popleft()
                # 변환된 프레임 적용
                trans_frame = apply_transform(trans, cur_frame, STD_SIZE)
                # 변환된 랜드마크 적용
                trans_landmarks = trans(q_landmarks.popleft())
                # 입술 패치 자르기
                sequence.append(cut_patch(trans_frame,
                                          trans_landmarks[start_idx:stop_idx],
                                          crop_height // 2,
                                          crop_width // 2))
            return np.array(sequence)  # 잘라낸 입술 패치 시퀀스 반환
        
        frame_idx += 1

    return None

def write_video_ffmpeg(rois: list[np.ndarray], target_path: str, ffmpeg: str) -> None:
    """
    주어진 이미지 시퀀스를 사용하여 비디오 파일을 생성합니다. 이미지들은 주어진 경로에 따라 임시 디렉토리에 저장된 후, 
    `ffmpeg`를 사용하여 비디오 파일로 병합됩니다.

    Args:
        rois (List[np.ndarray]): 비디오로 변환할 이미지들의 리스트. 각 이미지는 NumPy 배열로 표현됩니다.
        target_path (str): 생성할 비디오 파일의 저장 경로.
        ffmpeg (str): `ffmpeg` 실행 파일의 경로.

    Returns:
        None: 이 함수는 비디오를 생성하고 파일을 저장하는 작업만 수행합니다.
    """
    # 저장할 디렉토리가 없으면 생성
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    
    # 이미지 파일 이름을 위한 자릿수 설정
    decimals = 10
    fps = 25  # 비디오 프레임 속도 설정
    
    # 임시 디렉토리 생성
    tmp_dir = tempfile.mkdtemp()
    
    # 각 이미지 시퀀스를 임시 디렉토리에 저장
    for i_roi, roi in enumerate(rois):
        cv2.imwrite(os.path.join(tmp_dir, str(i_roi).zfill(decimals)+'.png'), roi)
    
    # 이미지 목록 파일 생성
    list_fn = os.path.join(tmp_dir, "list")
    with open(list_fn, 'w') as fo:
        fo.write("file " + "'" + tmp_dir + '/%0' + str(decimals) + 'd.png' + "'\n")
    
    # 비디오 파일이 이미 존재하면 삭제
    if os.path.isfile(target_path):
        os.remove(target_path)
    
    # ffmpeg 명령어 실행
    cmd = [ffmpeg, "-f", "concat", "-safe", "0", "-i", list_fn, "-q:v", "1", "-r", str(fps), '-y', '-crf', '20', target_path]
    pipe = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    
    # 임시 디렉토리 삭제
    shutil.rmtree(tmp_dir)
    
    return


## ====================  PROCESS  ====================
def preprocess_video(input_video_path: str,
                     output_video_path: str,
                     face_predictor_path: str,
                     mean_face_path: str):
    """
    비디오 파일에서 얼굴을 감지하고, 해당 얼굴의 랜드마크를 기준으로 얼굴 영역을 자르고, 
    얼굴을 정렬하여 비디오를 전처리하는 함수.

    주어진 입력 비디오 파일에서 얼굴을 감지하고, 랜드마크를 추출하여 얼굴 영역을 자른 후 
    표준 크기로 얼굴을 정렬하여 새로운 비디오를 생성. 이때 음성도 추출하여 저장.

    Args:
        input_video_path (str): 입력 비디오 파일의 경로.
        output_video_path (str): 출력 비디오 파일의 경로.
        face_predictor_path (str): dlib의 얼굴 랜드마크 예측기 모델 경로.
        mean_face_path (str): 평균 얼굴 랜드마크를 저장한 파일 경로.

    Returns:
        None
    """
    # 얼굴 감지기 및 랜드마크 예측기 설정
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(face_predictor_path)
    mean_face_landmarks = np.load(mean_face_path)
    
    # 비디오와 랜드마크 추출 설정
    STD_SIZE = (256, 256)
    stablePntsIDs = [33, 36, 39, 42, 45]
    video_path = os.path.join(input_video_path, output_video_path)
    videogen = skvideo.io.vread(video_path)
    frames = np.array([frame for frame in videogen])
    landmarks = []

    # 랜드마크 추출
    for frame in frames:
        landmark = detect_landmark(frame, detector, predictor)
        landmarks.append(landmark)

    # 랜드마크 보간
    preprocessed_landmarks = landmarks_interpolate(landmarks)

    # 얼굴 패치 자르기
    try:
        rois = crop_patch(
            video_pathname=video_path,
            landmarks=preprocessed_landmarks,
            mean_face_landmarks=mean_face_landmarks,
            stablePntsIDs=stablePntsIDs,
            STD_SIZE=STD_SIZE,
            window_margin=12,
            start_idx=48,
            stop_idx=68,
            crop_height=96,
            crop_width=96
        )
    except:
        print(f"Error processing video: {video_path}")
        return

    # 결과 저장
    roi_path = os.path.join(input_video_path, output_video_path[:-4] + '_roi.mp4')
    audio_fn = os.path.join(input_video_path, output_video_path[:-4] + '.wav')

    # 비디오 저장
    write_video_ffmpeg(rois, roi_path, "/usr/bin/ffmpeg")
    
    # 오디오 추출
    cmd = f"/usr/bin/ffmpeg -i {video_path} -f wav -vn -y {audio_fn} -loglevel quiet"
    subprocess.call(cmd, shell=True)
    return


## ====================   MAIN   ====================
if __name__ == "__main__":    

    PATH_TO_DIRECTORY = '../../data/'
    face_predictor_path = "../../misc/shape_predictor_68_face_landmarks.dat"
    mean_face_path = "../../misc/20words_mean_face.npy"

    to_iterate = list(os.walk(PATH_TO_DIRECTORY))
    count = 0
    for root, dirs, files in tqdm(to_iterate, total=len(to_iterate)):
        flag = False
        for file in files:
            if file.endswith('.mp4') and not file.endswith('_roi.mp4'):
                count += 1
                preprocess_video(input_video_path=root,
                                 output_video_path=file,
                                 face_predictor_path=face_predictor_path,
                                 mean_face_path=mean_face_path)