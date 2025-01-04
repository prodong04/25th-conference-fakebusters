import cv2
import os
import numpy as np
from tqdm import tqdm
import mediapipe as mp
from typing import Tuple
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarkerResult, FaceLandmarksConnections

class ROIProcessor:
    """
    """
    def __init__(self, video_path: str, config: dict) -> None:
        """
        Initializes ROIProcessor object, conducts landmark detection on input video upon initialization.

        Args:
            video_path: input video path
            config: configuration dictionary.
        """
        self.video_path = video_path
        self.landmarker_path = config["landmarker_path"]
        self.detection_result_list = []
        self.frame_list = []
        self.height = 0
        self.width = 0

        ## Set options for MediaPipe facelandmark detector
        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.landmarker_path),
            running_mode=VisionRunningMode.VIDEO)
        
        ## Loop through video frames for landmark detection
        with FaceLandmarker.create_from_options(options) as landmarker:
            cap = cv2.VideoCapture(self.video_path)
            ## We take the fps of the input video as "local" fps.
            ## We later interpolate this to fit 30 fps for ppg map.
            self.fps_local = int(cap.get(cv2.CAP_PROP_FPS))
            self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_idx = 0

            while cap.isOpened():
                success, frame = cap.read()

                if not success:
                    break
                
                ## Change frame color from BGR to RGB for later visualization.
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.height, self.width, _ = frame.shape
                ## Convert frame image into designated input type for MediaPipe model.
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                timestep_ms= int((frame_idx / self.fps_local) * 1000)
                detection_result = landmarker.detect_for_video(mp_image, timestep_ms)
                ## Append detection result and original frame image onto respective lists.
                self.detection_result_list.append(detection_result)
                # print('detection_result', detection_result)
                # breakpoint()
                self.frame_list.append(frame)
                frame_idx += 1

            ## Release resources.
            cap.release()
            cv2.destroyAllWindows()

    def crop_mouth(self, frame: np.ndarray, detection_result: FaceLandmarkerResult) -> np.ndarray:
        #print(frame)
        try:
            face_landmark = detection_result.face_landmarks[0]
        except IndexError:
            return np.zeros((128, 128, 3), dtype=np.float32)

        try:
            frame = np.copy(frame)
            landmark_points = [
                [int(landmark.x * self.width), int(landmark.y * self.height)]
                for landmark in face_landmark
            ]

            contains_negative = any(num < 0 for sublist in landmark_points for num in sublist)
            if contains_negative:
                raise Exception("Negative value detected in landmark points")
        except Exception as e:
            return np.zeros((128, 128, 3), dtype=np.float32)
        
        y1, y2 = landmark_points[0][1], landmark_points[17][1]
        x1, x2 = landmark_points[61][0], landmark_points[291][0]

        # Validate cropping range
        if y1 >= y2 or x1 >= x2:
            print(f"Invalid cropping range: y1={y1}, y2={y2}, x1={x1}, x2={x2}")
            breakpoint()
            return np.zeros((128, 128, 3), dtype=np.float32)
        if y1 < 0 or y2 > self.height or x1 < 0 or x2 > self.width:
            
            print(f"Cropping range out of bounds: y1={y1}, y2={y2}, x1={x1}, x2={x2}")
            return np.zeros((128, 128, 3), dtype=np.float32)

        # Perform cropping
        cropped_frame = frame[y1:y2, x1:x2]
        if cropped_frame.size == 0:
            
            print(f"Empty cropped frame with range: y1={y1}, y2={y2}, x1={x1}, x2={x2}")
            breakpoint()
            return np.zeros((128, 128, 3), dtype=np.float32)
        cropped_frame = cv2.resize(cropped_frame, (128, 128), interpolation=cv2.INTER_LINEAR)
        
        return cropped_frame

    def detect_with_crop(self) -> Tuple[np.ndarray, int]:

        cropped_frames = []
        with tqdm(total=self.frame_count, desc="Processing Frames", unit="frame") as pbar:
            for frame, detection_result in zip(self.frame_list, self.detection_result_list):
                # print('frame before crop',frame)
                # breakpoint()
                cropped_frame = self.crop_mouth(frame, detection_result)
                # Check if cropped_frame is a blank frame
                if np.all(cropped_frame == 0):
                    pbar.update(1)
                    continue
                cropped_frames.append(cropped_frame)
                pbar.update(1)
        return cropped_frames, self.fps_local

    def draw(self, frame: np.ndarray, detection_result: FaceLandmarkerResult) -> np.ndarray:
        """
        Visualizes a single frame.

        Args:
            frame: Video frame image.
            detection_result: Model inference results for the frame.

        Returns:
            annotated_frame: The frame with ROI polygons for the right, center, and left sides of the face visualized.
        """
        ## If no face is detected in the given frame, return the original frame as is.
        try:
            face_landmark = detection_result.face_landmarks[0]
        except IndexError:
            return frame
        
        ## Denormalize the coordinates of landmarks using the frame's height and width.
        annotated_frame = np.copy(frame)
        landmark_points = [
            (int(landmark.x * self.width), int(landmark.y * self.height))
            for landmark in face_landmark
        ]

        for connection in FaceLandmarksConnections.FACE_LANDMARKS_LIPS:
            start_coord = landmark_points[connection.start]
            end_coord = landmark_points[connection.end]
            polygon_points = np.array([start_coord, end_coord], dtype=np.int32)
            cv2.polylines(annotated_frame, [polygon_points], isClosed=True, color=(0, 255, 0), thickness=2)
        return annotated_frame

    def detect_with_draw(self) -> Tuple[np.ndarray, int]:
        """
        Visualizes the boundary lines of the face center, left cheek, and right cheek ROIs 
        detected by the Mediapipe FaceLandmark model.

        Returns:
            annotated_frames: An array of frames with visual annotations.
            fps_local: The frames per second (FPS) used for the video.
        """
        annotated_frames = np.empty((self.frame_count, self.height, self.width, 3), dtype=np.uint8)
        with tqdm(total=self.frame_count, desc="Processing Frames", unit="frame") as pbar:
            for index, (frame, detection_result) in enumerate(zip(self.frame_list, self.detection_result_list)):
                annotated_frames[index] = self.draw(frame, detection_result)
                pbar.update(1)
        return annotated_frames, self.fps_local
    
    def save_cropped_video(self, cropped_frames: np.ndarray, fps: int) -> None:
        """
        Saves the cropped frames as a video file.

        Args:
            cropped_frames: List of cropped frames (H, W, C) format.
            fps: Frames per second for the output video.
        """

        # Get output file name by appending '_cropped' to the original video name
        base_name = os.path.splitext(os.path.basename(self.video_path))[0]
        output_dir = os.path.dirname(self.video_path)
        output_path = os.path.join(output_dir, f"{base_name}_cropped.mp4")

        # Get frame size from the first frame
        height, width = cropped_frames[0].shape[:2]
        channels = 1 if len(cropped_frames[0].shape) == 2 else cropped_frames[0].shape[2]

        # Initialize VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
        is_color = channels == 3
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height), is_color)

        if not video_writer.isOpened():
            raise ValueError(f"VideoWriter failed to open with path: {output_path}")

        # Write each frame to the video
        for idx, frame in enumerate(cropped_frames):
            # Ensure the frame is in uint8 format
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)  # Normalize and convert to uint8 if needed

            if channels == 1:
                # Grayscale to BGR conversion for compatibility with VideoWriter
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif channels == 3:
                # Check if the frame is likely in RGB format by analyzing channel order
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Validate frame size
            if frame.shape[:2] != (height, width):
                raise ValueError(f"Frame {idx} has an unexpected size: {frame.shape[:2]}, expected: {(height, width)}")

            video_writer.write(frame)

        # Release the video writer
        video_writer.release()
        return output_path










