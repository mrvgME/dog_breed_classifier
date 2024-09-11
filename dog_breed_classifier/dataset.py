
import os
from pathlib import WindowsPath
from typing import Optional

import numpy as np
from glob import glob

import cv2

from dotenv import load_dotenv

load_dotenv()


class LoadImageData:

    def __init__(self, data_path: Optional[WindowsPath]):
        """
        Define data paths and folder names.

        Parameter:
        data_path: Path to the data.
        """

        self.human_files = "lfw"
        self.dog_files = "dog_images"

        __project_dir = os.getenv("project_dir")

        if data_path is None:
            data_path = WindowsPath("data/raw/")

        self.file_path = os.path.join(__project_dir, data_path)

    def read_dataset(self, dataset_name: str):
        """
        Create numpy array with paths to images.

        Parameter:
        dataset_name: Path to the data. humans or dogs.
        """

        if dataset_name not in ['humans', 'dogs']:
            raise ValueError(
                "{} is not implemented. Implemented files are {}".format(
                    dataset_name, ['humans', 'dogs']
                )
            )

        if dataset_name == 'humans':
            files = np.array(
                glob(
                    os.path.join(self.file_path, self.human_files) + "/*/*"
                )
            )
        elif dataset_name == 'dogs':
            files = np.array(
                glob(
                    os.path.join(self.file_path, self.dog_files) + "/*/*/*"
                )
            )

        return files
    
    def read_image(self, image_path: str):
        """
        Read image given its path using opencv.

        Parameter:
        image_path: Path to the image.
        """

        img = cv2.imread(image_path)

        return img
    
    def convert_BGR2GRAY(self, img: cv2):
        """
        Convert BGR image to grayscale

        Parameter:
        img: Image opened with opencv.
        """

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        return gray_img
        

if __name__ == "__main__":
    DataLoader = LoadImageData(None)
    human_files = DataLoader.read_dataset('humans')
