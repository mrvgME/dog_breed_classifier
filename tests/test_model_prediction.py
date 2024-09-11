
import unittest
from dog_breed_classifier.dataset import LoadImageData
from dog_breed_classifier.modelling.predict import LoadModelData


class TestModelPrediction(unittest.TestCase):

    def test_face_prediction(self):
        """
        Check if read dataset works.
        """

        ModelLoader = LoadModelData(None)
        DataLoader = LoadImageData(None)

        # Read image and convert to grayscale
        human_files = DataLoader.read_dataset('humans')
        img = DataLoader.read_image(human_files[10])
        gray_img = DataLoader.convert_BGR2GRAY(img)

        # Face prediction
        _, face_prediction = ModelLoader.predict_pre_trained_model('frontalface', gray_img)

        self.assertTrue(len(face_prediction) == 1)

