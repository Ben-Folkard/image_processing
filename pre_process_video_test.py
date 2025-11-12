import cv2
import numpy as np
import imageio.v3 as iio
import os
from time import perf_counter as time
from shutil import rmtree

DTYPE_IMAGE = np.uint8


def pre_process_video_frames(filename, output_folder, step_size, num_frames, grayscale=True):
    frame_output_folder = f"{output_folder}/{filename[:-4]}_frames"
    os.makedirs(frame_output_folder, exist_ok=True)
    start = 0
    frames = []
    for i, frame in enumerate(iio.imiter(filename)):
        if grayscale and frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame)

        if ((i + 1) % step_size == 0) or (i == num_frames - 1):
            np.save(f"{frame_output_folder}/{start} -> {i}", np.array(frames, dtype=DTYPE_IMAGE))
            start = i + 1
            frames = []


def load_video_frames(filename, output_folder, frames_start, frames_end):
    return np.load(f"{output_folder}/{filename[:-4]}_frames/{frames_start} -> {frames_end}.npy")


def delete_pre_processed_video_frames(filename, output_folder):
    rmtree(f"{output_folder}/{filename[:-4]}_frames")


if __name__ == "__main__":
    filename = "11_01_H_170726081325.avi"
    output_folder = "results"
    step_size = 1000

    cap = cv2.VideoCapture(filename)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print("Running...")
    start = time()
    pre_process_video_frames(filename, output_folder, step_size, num_frames)
    print(f"Pre-processed in {time()-start}s")
    print(load_video_frames(filename, output_folder, 0, step_size-1))
