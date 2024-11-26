import cv2
import numpy as np
import os
def calculate_histogram_diff(frame1, frame2):
    """
    Calculate the histogram difference between two frames.
    """
    # Convert frames to grayscale
    gray_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Calculate the histogram for both frames
    hist1 = cv2.calcHist([gray_frame1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([gray_frame2], [0], None, [256], [0, 256])

    # Normalize histograms
    hist1 = cv2.normalize(hist1, hist1)
    hist2 = cv2.normalize(hist2, hist2)

    # Calculate histogram difference
    diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)

    return diff

def detect_transitions(video_path, threshold, fps):
    """
    Detect transitions in the video using histogram differences.
    """
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    ret, prev_frame = cap.read()
    frame_number = 0
    transitions = []

    while ret:
        ret, curr_frame = cap.read()
        if not ret:
            break

        diff = calculate_histogram_diff(prev_frame, curr_frame)

        if diff > threshold:
            print(frame_number)
            transitions.append(float((frame_number-1)/fps))

        prev_frame = curr_frame
        frame_number += 1

    cap.release()

    return transitions

if __name__ == "__main__":
    video_path = '/data/fanshen/workspace/preprocessing/test_video/'
    for video in os.listdir(video_path):
        transitions = detect_transitions(os.path.join(video_path, video), 0.5, 25)
        print(video)
        print("Scene changes detected at frames:", transitions)
