import cv2
import os
import numpy as np
import csv
import re
from timeit import default_timer as timer
from datetime import datetime


# Class for handling the reading of dataset
class DatasetDirectoryHandler:
    def __init__(self, base_folder):
        self.base_folder = base_folder

    def get_subject_folders(self):
        return self._get_subfolders(self.base_folder)

    def get_activity_folders(self, subject_folder):
        subject_path = os.path.join(self.base_folder, subject_folder)
        return self._get_subfolders(subject_path)

    def get_trial_folders(self, subject_folder, activity_folder):
        activity_path = os.path.join(self.base_folder, subject_folder, activity_folder)
        return self._get_subfolders(activity_path)

    def get_camera_folders(self, subject_folder, activity_folder, trial_folder):
        trial_path = os.path.join(
            self.base_folder, subject_folder, activity_folder, trial_folder
        )
        return self._get_subfolders(trial_path)

    def _get_subfolders(self, folder):
        folders = [
            d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))
        ]
        return sorted(folders, key=lambda x: int(re.search(r"\d+", x).group()))


# Class for loading frames
class FrameLoader:
    def __init__(self, dataset_folder):
        self.dataset_folder = dataset_folder

    def get_video_folders(self):
        return [
            d
            for d in os.listdir(self.dataset_folder)
            if os.path.isdir(os.path.join(self.dataset_folder, d))
        ]

    def load_frames_from_video(self, video_folder):
        image_folder = os.path.join(self.dataset_folder, video_folder)
        file_names = sorted(
            [
                f
                for f in os.listdir(image_folder)
                if f.endswith(".jpg") or f.endswith(".png")
            ]
        )

        return [
            (fn[:-4], cv2.imread(os.path.join(image_folder, fn), cv2.IMREAD_GRAYSCALE))
            for fn in file_names
        ]


# defines the background Subtraction Model
class bg_sub:
    def __init__(self):
        self.fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        self.fgbg.setShadowValue(0)
        self.fgbg.setShadowThreshold(0.5)

    # Function to perform edge Detection
    def compute_edge_detector(self, current_frame):
        current_frame = cv2.normalize(
            src=current_frame,
            dst=None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_8U,
        )
        fgmask_curr = self.fgbg.apply(current_frame)
        kernel = np.ones((5, 5), np.uint8)
        fgmask_curr = cv2.morphologyEx(fgmask_curr, cv2.MORPH_OPEN, kernel)
        fgmask_curr = cv2.morphologyEx(fgmask_curr, cv2.MORPH_CLOSE, kernel)
        current_frame_masked = cv2.bitwise_and(
            current_frame, current_frame, mask=fgmask_curr
        )
        current_frame_masked = cv2.Canny(current_frame_masked, 65, 195)
        resized_frame = cv2.resize(current_frame_masked, (51, 38))
        return resized_frame


# class for writing the data into array
class NumpyWriter:
    def __init__(self, output_folder):
        self.output_folder = output_folder
        os.makedirs(output_folder, exist_ok=True)

    def write_array(self, array, name):
        file_path = os.path.join(self.output_folder, f"{name}.npy")
        directory = os.path.dirname(file_path)
        os.makedirs(directory, exist_ok=True)
        np.save(file_path, array)


# Class to process Optical Flow
class OpticalFlowProcessor:
    def __init__(self, dataset_folder, output_folder, fps=18):
        self.frame_loader = FrameLoader(dataset_folder)
        self.numpy_writer = NumpyWriter(output_folder)
        self.fps = fps
        self.window_size = fps
        self.overlap = fps // 2
        self.bg_subtraction = bg_sub()

    def total_seconds_from_timestamp(timestamp: str) -> float:
        hours, minutes, seconds = map(float, timestamp.split("T")[1].split("_"))
        total_seconds = hours * 3600 + minutes * 60 + seconds
        return total_seconds

    def increment_timestamp(timestamp: str) -> str:
        date, time = timestamp.split("T")
        try:
            hours, minutes, remainder = time.split("_")
            seconds, ms = remainder.split(".")
        except ValueError:
            print(f"Error with timestamp: {time}")
            raise

        ms = int(ms)
        seconds = int(seconds)
        minutes = int(minutes)
        hours = int(hours)

        ms += 500000
        if ms >= 1000000:
            ms -= 1000000
            seconds += 1

        if seconds >= 60:
            seconds -= 60
            minutes += 1

        if minutes >= 60:
            minutes -= 60
            hours += 1

        time_str = f"{hours:02}_{minutes:02}_{seconds:02}.{ms:06}"
        return f"{date}T{time_str}"

    # procedure to process videos
    def process_video(self, video_folder):
        frames = self.frame_loader.load_frames_from_video(video_folder)
        i = 0
        num_frames_in_window = int(self.fps)
        overlap_frames = int(self.overlap)
        last_frame_time = frames[-1][0]
        last_frame_seconds = OpticalFlowProcessor.total_seconds_from_timestamp(
            last_frame_time
        )
        timestamp = frames[i][0]
        while i < len(frames) - num_frames_in_window:
            window_end = min(i + num_frames_in_window, len(frames))
            # optical_flows_u= []
            # optical_flows_v = []
            canny_frames = []
            for j in range(i, window_end - 1):
                try:
                    final_components = self.bg_subtraction.compute_edge_detector(
                        frames[j + 1][1]
                    )
                    canny_frames.append(final_components)
                except cv2.error as e:
                    print(
                        f"Error processing frame {frames[j][0]} from video {video_folder}. Error: {e}"
                    )
                    continue
            components_stacked = np.stack(canny_frames, axis=0)
            window_name = f"{video_folder}_{timestamp}"
            self.numpy_writer.write_array(components_stacked, window_name)
            timestamp = OpticalFlowProcessor.increment_timestamp(timestamp)
            next_increment_seconds = OpticalFlowProcessor.total_seconds_from_timestamp(
                timestamp
            )
            if last_frame_seconds - next_increment_seconds < 1.0:
                break
            i += num_frames_in_window - overlap_frames

    def run(self):
        dir_handler = DatasetDirectoryHandler(self.frame_loader.dataset_folder)
        logging_output(f"# Data Process starts for {output_folder} ##", log_path)
        for subject_folder in dir_handler.get_subject_folders():
            for activity_folder in dir_handler.get_activity_folders(subject_folder):
                for trial_folder in dir_handler.get_trial_folders(
                    subject_folder, activity_folder
                ):
                    for camera_folder in dir_handler.get_camera_folders(
                        subject_folder, activity_folder, trial_folder
                    ):
                        print(f"Processing video: {camera_folder}")
                        logging_output(f"Processing video: {camera_folder},log_path")

                        start = timer()
                        self.process_video(
                            os.path.join(
                                subject_folder,
                                activity_folder,
                                trial_folder,
                                camera_folder,
                            )
                        )
                        print("Time Taken: ", timer() - start)
                        logging_output(f"Time Taken: {timer() - start}", log_path)
                        now = datetime.now()
                        current_time = now.strftime("%H:%M:%S")
                        print("Process Completed at : ", current_time)


def logging_output(message, file_path=f"../Outputs/prep_log.txt"):
    try:
        # Write to file
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        with open(file_path, "a") as file:
            file.write(f"{current_time} : {message} \n")
    except Exception as e:
        print(f"Error logging: {e}")


if __name__ == "__main__":
    try:
        model_type = "Exp_1_1_OG_BGS_Canny"
        # To print the processing logs
        log_path = f"../Outputs/{model_type}/prep_log.log"
        dataset_folder = "../UP-Fall"
        output_folder = f"../Outputs/{model_type}/Unbalanced"
        processor = OpticalFlowProcessor(dataset_folder, output_folder)
        processor.run()
        print("###############Data Process Complete############")
        logging_output(f"# Data Process Complete for {output_folder} ##", log_path)
        print("~~~~~~~~Entering Hibernation Mode ~~~~~~~~~~~~~")
    except:
        print("~~~~~~~~Error Occurred~~~~~~~~")
