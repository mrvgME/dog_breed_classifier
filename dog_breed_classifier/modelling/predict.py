
import os
from pathlib import WindowsPath
from typing import Optional

import numpy as np

import cv2  

from dotenv import load_dotenv
load_dotenv()

from dog_breed_classifier.dataset import LoadImageData


class LoadModelData:

    def __init__(self, model_path: Optional[WindowsPath]):
        """
        Define data paths and folder names

        Parameter:
        model_path: Path to the pre-trained model.
        """

        self.human_detector = "haarcascade_frontalface_alt.xml"

        __project_dir = os.getenv("project_dir")

        if model_path is None:
            model_path = WindowsPath("models/haarcascades/")

        self.pre_trained_model_path = os.path.join(__project_dir, model_path)

    def predict_pre_trained_model(self, model_name: str, img: str):
        """
        Code to extract and predict pre-trained model.
        Human face detector pre-trained models downloaded from: 
        https://github.com/opencv/opencv/tree/master/data/haarcascades

        Dog breed detector pre-trained model from 



        Parameter:
        model_name: Name of the model. frontalface.
        """

        __implemented_models__ = ['frontalface']

        if model_name not in __implemented_models__:
            raise ValueError(
                "{} is not implemented. Implemented models are {}".format(
                    model_name, __implemented_models__
                )
            )

        if model_name == 'frontalface':
            model_path_ = os.path.join(
                self.pre_trained_model_path, self.human_detector
            )
            model_ = cv2.CascadeClassifier(model_path_)
            prediction = model_.detectMultiScale(img)

        return model_, prediction
        

if __name__ == "__main__":
    ModelLoader = LoadModelData(None)
    DataLoader = LoadImageData(None)
    human_files = DataLoader.read_dataset('humans')
    img = DataLoader.read_image(human_files[10])
    gray_img = DataLoader.convert_BGR2GRAY(img)
    human_model, face_prediction = ModelLoader.predict_pre_trained_model('frontalface', gray_img)
