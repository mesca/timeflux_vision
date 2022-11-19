from timeflux.core.node import Node
import numpy as np
import cv2
import mediapipe as mp


class HandPose(Node):

    def __init__(self, device=0):
        self.cap = cv2.VideoCapture(device)
        self._names = ['x_thumb', 'y_thumb', 'x_index', 'y_index',
                         'x_middle', 'y_middle', 'x_ring', 'y_ring',
                         'x_pinky', 'y_pinky']

        if not self.cap.isOpened():
            print("Cannot open camera")
            exit()
        self.mp_hands = mp.solutions.hands

    def update(self):
        with self.mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
            success, image = self.cap.read()
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                self.o.set(
                        np.array([hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP].x,
                            hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP].y,
                            hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].x,
                            hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y, 
                            hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x,
                            hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y, 
                            hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP].x,
                            hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP].y,
                            hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP].x,
                            hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP].y]).reshape(1,10),
                            names=self._names,
                    )

            
