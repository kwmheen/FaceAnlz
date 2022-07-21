from .FaceAnlz import FaceAnlz  # Inherits FaceAnlz

import cv2
import mediapipe as mp
from datetime import datetime
import pandas as pd


class FaceDetection(FaceAnlz):
    def __init__(
        self, file_dir: str, min_detection_confidence: float = 0.5, save_dir: str = ""
    ):
        super().__init__(file_dir, min_detection_confidence, save_dir)
        self.confrim_setting()

    def confrim_setting(self):
        """Set MediaPipe Variables setting from Attribute api_info"""
        self.mp_face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=self.api_info["model_selection"],
            min_detection_confidence=self.api_info["min_detection_confidence"],
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def get_eye_coord(
        self,
        show_process: bool = False,
        expansion_rate: float = 1,
        save_csv: bool = False,
    ) -> list:
        """Get eye coordinate of source file.

        Args:
            show_process (bool) : Wether show process of detecting the source file. Default False
            expansion_rate (float) : Source file expansion rate. Default 1
            save_csv (bool) : Save Result as CSV in current directory. Default False

        Returns:
            list of dict. Contains location of eye, fps/frame if source file is video.

        """
        if self.api_info != self._INITIAL_API_INFO:
            self.confrim_setting()

        if self.file_type == "image":
            return self.__get_image_eye_coord(show_process, expansion_rate, save_csv)
        elif self.file_type == "video":
            return self.__get_video_eye_coord(show_process, expansion_rate, save_csv)

    def __get_video_eye_coord(
        self,
        show_process: bool = False,
        expansion_rate: float = 1,
        save_csv: bool = False,
    ) -> list:
        """Get eye coordinate of source file, which is video.

        Args:
            show_process (bool) : Wether show process of detecting the source file. Default False
            expansion_rate (float) : Source file expansion rate. Default 1
            save_csv (bool) : Save Result as CSV in video directory. Default False

        Returns:
            eye_tracking_list (list) : list of dict. Contains location of eye, fps/frame

        """
        eye_tracking_list = []
        dt = str(datetime.now())

        while self.source.isOpened():
            status, image = self.source.read()
            if status:
                image.flags.writeable = False  # To improve performance, optianlly mark the image as not writeable
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = self.mp_face_detection.process(image)

                coord_info = self.detection_to_eye_coordinate(
                    results
                )  # records video frame, fps
                for x in coord_info:
                    x["frame"] = self.source.get(cv2.CAP_PROP_POS_FRAMES)
                    x["fps"] = self.source.get(cv2.CAP_PROP_FPS)
                eye_tracking_list = eye_tracking_list + coord_info

                if show_process:
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    if results.detections:

                        for detection in results.detections:
                            self.mp_drawing.draw_detection(image, detection)
                    image = cv2.resize(
                        image,
                        (
                            int(image.shape[0] * expansion_rate),
                            int(image.shape[1] * expansion_rate),
                        ),
                    )
                    cv2.imshow(dt, image)
                    if cv2.waitKey(5) & 0xFF == 27:
                        break
            else:
                break
        self.source.release()
        cv2.destroyAllWindows()

        if save_csv:
            df = pd.DataFrame(eye_tracking_list)
            df.to_csv(
                self.save_dir + "/" + self.file_name + "_eye_coord_detection.csv",
                index=False,
            )
            print(
                "SAVED : "
                + self.save_dir
                + "/"
                + self.file_name
                + "_eye_coord_detection.csv"
            )

        return eye_tracking_list

    def __get_image_eye_coord(
        self,
        show_process: bool = False,
        expansion_rate: float = 1,
        save_csv: bool = False,
    ) -> list:
        """Get eye coordinate of source file, which is image.

        Args:
            show_process (bool) : Wether show process of detecting the source file. Default False
            expansion_rate (float) : Source file expansion rate. Default 1
            save_csv (bool) : Save Result as CSV in image directory. Default False

        Returns:
            coord_info (list) : list of dict. Contains location of eye

        """
        results = self.mp_face_detection.process(
            cv2.cvtColor(self.source, cv2.COLOR_BGR2RGB)
        )
        coord_info = self.detection_to_eye_coordinate(results)
        if results.detections:
            annotated_image = self.source.copy()

            if show_process:
                for detection in results.detections:
                    self.mp_drawing.draw_detection(annotated_image, detection)
                annotated_image = cv2.resize(
                    annotated_image,
                    (
                        int(annotated_image.shape[0] * expansion_rate),
                        int(annotated_image.shape[1] * expansion_rate),
                    ),
                )
                # DEPRECATED FUNCTION
                # Because get_video_coord does not suppports Save function, deprecated.
                """
                if save_img:
                    annotated_dir = '/'.join(self.file_dir.split('/')[:-1]) + '/Annotated_' + self.file_dir.split('/')[-1]
                    cv2.imwrite(annotated_dir, annotated_image) #Save 'Example.png' to 'Annotated_Example.png'
                """
                cv2.imshow(str(datetime.now()), annotated_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        else:
            print("Cannot Recognize anny face")
        return coord_info  # TODO return type 정하기

    def detection_to_eye_coordinate(self, results):
        """Convert FaceDetection result to eye coorindate list

        Args:
            process_result  (FaceDetection.process()) : Process of FaceDetection

        Returns:
            coord_list (list) : eye coordinate list that contains location, subject, x, y, z coordinate

        """
        if results.detections:  # If Something Has Detected
            return_list = []
            subject_count = len(results.detections)
            for subject_id in range(subject_count):
                detect = results.detections
                mp_face_detection = mp.solutions.face_detection
                eye_left = mp_face_detection.get_key_point(
                    detect[subject_id], mp_face_detection.FaceKeyPoint.LEFT_EYE
                )
                eye_right = mp_face_detection.get_key_point(
                    detect[subject_id], mp_face_detection.FaceKeyPoint.RIGHT_EYE
                )
                return_list += [
                    {"x": eye_left.x, "y": eye_left.y, "loc": "left"},
                    {"x": eye_right.x, "y": eye_right.y, "loc": "right"},
                ]
            return return_list
        else:  # If Nothing Has Detected
            return [
                {"x": None, "y": None, "loc": "left"},
                {"x": None, "y": None, "loc": "right"},
            ]
