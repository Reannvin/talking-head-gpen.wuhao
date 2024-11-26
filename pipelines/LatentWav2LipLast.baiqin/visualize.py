import cv2
import numpy as np


def visualize_landmark_image(frame,landmark):
    for i in range(landmark.shape[0]):
        cv2.circle(frame, (int(landmark[i][0]), int(landmark[i][1])), 1, (0, 0, 255), -1)
    return frame

def visualize_landmark_video(frames,landmarks):
    for i in range(len(frames)):
        frames[i]=visualize_landmark_image(frames[i],landmarks[i])
    return frames

