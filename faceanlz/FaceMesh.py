from .FaceAnlz import FaceAnlz  # Inherits FaceAnlz

import cv2
import mediapipe as mp
from datetime import datetime
import pandas as pd


class FaceMesh(FaceAnlz):
    def __init__(
        self, file_dir: str, min_detection_confidence: float = 0.5, save_dir: str = ""
    ):
        super().__init__(file_dir, min_detection_confidence, save_dir)
        self.confrim_setting()
        self.FACEMESH_EYE = {"left": [468], "right": [473]}
        self.FACEMESH_EYEBROW = {
            "left": [334, 276, 293, 295, 296, 282, 283, 300, 265, 336],
            "right": [46, 66, 52, 53, 63, 70, 65, 105, 55, 107],
        }
        self.FACEMESH_EYEBROW_TEST = {'left':[46],'right':[334]}

    def confrim_setting(self):
        """Set MediaPipe Variables setting from Attribute api_info"""
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=self.api_info["static_image_mode"],
            max_num_faces=self.api_info["max_num_faces"],
            refine_landmarks=self.api_info[
                "refine_landmarks"
            ],  # DO NOT FIX IT -> Coordinate 좌표가 바뀌게 됨
            min_detection_confidence=self.api_info["min_detection_confidence"],
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def get_coord(
        self,
        coord_info: dict,
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
            return self.__get_image_coord(
                coord_info, show_process, expansion_rate, save_csv
            )
        elif self.file_type == "video":
            return self.__get_video_coord(
                coord_info, show_process, expansion_rate, save_csv
            )

    def __get_video_coord(
        self,
        coord_info: dict,
        show_process: bool,
        expansion_rate: float,
        save_csv: bool = False,
    ):
        """Get eye coordinate of source file, which is video.

        Args:
            show_process (bool) : Wether show process of detecting the source file. Default False
            expansion_rate (float) : Source file expansion rate. Default 1
            save_csv (bool) : Save Result as CSV in Video directory. Default False

        Returns:
            eye_tracking_list (list) : list of dict. Contains location of eye, fps/frame

        """
        # Let's think step by step
        eye_tracking_list = []
        mp_face_mesh = mp.solutions.face_mesh
        dt = str(datetime.now())

        while self.source.isOpened():
            status, image = self.source.read()
            if status:
                image.flags.writeable = False  # To improve performance, optianlly mark the image as not writeable
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = self.mp_face_mesh.process(image)

                mesh_info = self.mesh_to_coordinate(results, coord_info)
                for x in mesh_info:  # records video frame, fps
                    x["frame"] = self.source.get(cv2.CAP_PROP_POS_FRAMES)
                    x["fps"] = self.source.get(cv2.CAP_PROP_FPS)
                eye_tracking_list = eye_tracking_list + mesh_info

                if show_process:
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    if results.multi_face_landmarks:
                        for face_landmarks in results.multi_face_landmarks:
                            self.mp_drawing.draw_landmarks(
                                image=image,
                                landmark_list=face_landmarks,
                                connections=mp_face_mesh.FACEMESH_TESSELATION,
                                landmark_drawing_spec=None,
                                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style(),
                            )
                            self.mp_drawing.draw_landmarks(
                                image=image,
                                landmark_list=face_landmarks,
                                connections=mp_face_mesh.FACEMESH_CONTOURS,
                                landmark_drawing_spec=None,
                                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style(),
                            )
                            self.mp_drawing.draw_landmarks(
                                image=image,
                                landmark_list=face_landmarks,
                                connections=mp_face_mesh.FACEMESH_IRISES,
                                landmark_drawing_spec=None,
                                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_iris_connections_style(),
                            )
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
                self.save_dir + "/" + self.file_name + "_coord_mesh.csv", index=False
            )
            print("SAVED : " + self.save_dir + "/" + self.file_name + "_coord_mesh.csv")
        return eye_tracking_list

    def __get_image_coord(
        self,
        coord_info: dict,
        show_process: bool,
        expansion_rate: float,
        save_csv: bool = False,
    ):
        """Get eye coordinate of source file, which is image.

        Args:
            show_process (bool) : Wether show process of detecting the source file. Default False
            expansion_rate (float) : Source file expansion rate. Default 1
            save_csv (bool) : Save Result as CSV in Image directory. Default False

        Returns:
            mesh_info (list) : list of dict. Contains location of eye

        """
        mp_face_mesh = mp.solutions.face_mesh
        results = self.mp_face_mesh.process(
            cv2.cvtColor(self.source, cv2.COLOR_BGR2RGB)
        )
        mesh_info = self.mesh_to_coordinate(results, coord_info)

        image = self.source.copy()
        if show_process:
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    self.mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style(),
                    )
                    self.mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style(),
                    )
                    self.mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_iris_connections_style(),
                    )
                image = cv2.resize(
                    image,
                    (
                        int(image.shape[0] * expansion_rate),
                        int(image.shape[1] * expansion_rate),
                    ),
                )
                cv2.imshow(str(datetime.now()), image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        else:
            print("Cannot Recognize anny face")
        return mesh_info  # TODO return type 정하기

    def OLD_mesh_to_eye_coordinate(self, results):
        """Convert FaceMesh result to eye coorindate list

        Args:
            process_result  (FaceMesh.process()) : Process of FaceMesh

        Returns:
            coord_list (list) : eye coordinate list that contains location, subject, x, y, z coordinate

        """
        if results.multi_face_landmarks:
            coord_list = []
            for idx, face_landmark in enumerate(results.multi_face_landmarks):
                coord_list += [
                    {
                        "loc": "left",
                        "x": face_landmark.landmark[468].x,
                        "y": face_landmark.landmark[468].y,
                        "z": face_landmark.landmark[468].z,
                    },
                    {
                        "loc": "right",
                        "x": face_landmark.landmark[473].x,
                        "y": face_landmark.landmark[473].y,
                        "z": face_landmark.landmark[473].z,
                    },
                ]
            return coord_list
        else:
            return [
                {"loc": "left", "x": None, "y": None, "z": None},
                {"loc": "right", "x": None, "y": None, "z": None},
            ]

    def mesh_to_coordinate(self, results, coord_info):
        """Convert FaceMesh result to eye coorindate list

        Args:
            process_result  (FaceMesh.process()) : Process of FaceMesh

        Returns:
            coord_list (list) : eye coordinate list that contains location, subject, x, y, z coordinate

        """
        if results.multi_face_landmarks:
            coord_list = []
            for idx, face_landmark in enumerate(results.multi_face_landmarks):
                coord_list += [
                    {
                        "loc": "left",
                        "x": sum(
                            [
                                float(face_landmark.landmark[coord].x)
                                for coord in coord_info["left"]
                            ]
                        )
                        / len(coord_info["left"]),
                        "y": sum(
                            [
                                float(face_landmark.landmark[coord].y)
                                for coord in coord_info["left"]
                            ]
                        )
                        / len(coord_info["left"]),
                        "z": sum(
                            [
                                float(face_landmark.landmark[coord].z)
                                for coord in coord_info["left"]
                            ]
                        )
                        / len(coord_info["left"]),
                    },
                    {
                        "loc": "right",
                        "x": sum(
                            [
                                float(face_landmark.landmark[coord].x)
                                for coord in coord_info["right"]
                            ]
                        )
                        / len(coord_info["right"]),
                        "y": sum(
                            [
                                float(face_landmark.landmark[coord].y)
                                for coord in coord_info["right"]
                            ]
                        )
                        / len(coord_info["right"]),
                        "z": sum(
                            [
                                float(face_landmark.landmark[coord].z)
                                for coord in coord_info["right"]
                            ]
                        )
                        / len(coord_info["right"]),
                    },
                ]
            return coord_list
        else:
            return [
                {"loc": "left", "x": None, "y": None, "z": None},
                {"loc": "right", "x": None, "y": None, "z": None},
            ]
