import cv2
import mediapipe as mp
from os.path import exists
from datetime import datetime


class FaceAnlz:
    def __init__(
        self, file_dir: str, min_detection_confidence: float = 0.5, save_dir: str = ""
    ):
        # Set MedaiPipe Default Info
        self.api_info = {
            "min_detection_confidence": 0.5,  # Confidence Threshold
            "model_selection": 0,  # 0 means recognizing close face (within 5m), 1 means the opposite.
            "static_image_mode": True,  # Whether Image is Static or Not.
            "max_num_faces": 1,  # Maximum number of face to detect.
            "refine_landmarks": True,  # Needs when refining landmarks for lips and eyebrow.
            "min_tracking_confidence": 0.5,  # Higer Value can increase robustness of the solution & latency.
        }
        self._INITIAL_API_INFO = self.api_info

        # Check Whether File Exists
        self.file_dir = file_dir
        if not exists(self.file_dir):
            raise Exception("No Such File Exists. Check File directory.")

        # Check File Type
        file_extension = self.file_dir.split(".")[-1]
        file_classification = {
            "mov": "video",
            "mp4": "video",
            "avi": "video",
            "wmv": "video",
            "mkv": "video",
            "png": "image",
            "jpeg": "image",
            "jpg": "image",
            "raw": "image",
        }
        if file_extension not in file_classification:
            raise Exception(
                "Unprocessable file extensions. Check if it is video or image file."
            )
        self.file_type = file_classification[file_extension]
        if self.file_type == "video":
            self.source = cv2.VideoCapture(self.file_dir)
            self.api_info["static_image_mode"] = False
        elif self.file_type == "image":
            self.source = cv2.imread(self.file_dir, cv2.IMREAD_COLOR)
            self.api_info["static_image_mode"] = True

        # Set API info
        self.api_info["min_detection_confidence"] = min_detection_confidence

        # Set Dir Where To Save CSV
        self.file_name = (".".join(self.file_dir.split(".")[:-1])).split("/")[-1]
        if save_dir:
            self.save_dir = save_dir
        else:
            self.save_dir = "/".join(self.file_dir.split("/")[:-1])

    def get_eye_coord(
        self,
        show_process: bool = False,
        expansion_rate: float = 1,
        save_csv: bool = False,
    ) -> list:
        raise Exception("Struct Function For FaceDetection, FaceMesh")

    def confrim_setting(
        self,
        show_process: bool = False,
        expansion_rate: float = 1,
        save_csv: bool = False,
    ) -> list:
        raise Exception("Struct Function For FaceDetection, FaceMesh")
