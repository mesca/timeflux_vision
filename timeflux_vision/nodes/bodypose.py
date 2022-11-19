from timeflux.core.node import Node
import numpy as np
import mediapipe as mp
import cv2


class BodyPose(Node):
    """

    Attributes:
        i_lf (Port): Low Frequency power, expects DataFrame.
        i_hf (Port): High Frequency power, expects DataFrame.
        o (Port): Default output, provides DataFrame with column 'lf', 'hf', and 'lf/hf' and meta.
    """

    def __init__(self, device=0):
        self._names = ['x_nose', 'y_nose', 'x_l_elbow', 'y_l_elbow', 'x_r_elbow', 'y_r_elbow', 'x_l_hip', 'y_l_hip', 'x_r_hip', 'y_r_hip', 'x_l_foot', 'y_l_foot', 'x_r_foot', 'y_r_foot']
        self.cap = cv2.VideoCapture(device)
        self.mp_pose = mp.solutions.pose

    def update(self):
        with self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
            success, image = self.cap.read()
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            if results.pose_landmarks:
                pose_landmarks = results.pose_landmarks
                self.o.set(
                        np.array([pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE].x,
                                pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE].y,
                                pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW].x,
                                pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW].y, 
                                pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW].x,
                                pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW].y, 
                                pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP].x,
                                pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP].y,
                                pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP].x,
                                pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP].y,
                                pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x,
                                pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y,
                                pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x,
                                pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y]).reshape(1,14),
                            names=self._names,
                    )

        
