import unittest
from dog_breed_classifier.dataset import LoadImageData


class TestDataLoader(unittest.TestCase):

    def test_read_dataset(self):
        """
        Check if read dataset works.
        """

        DataLoader = LoadImageData(None)
        human_files = DataLoader.read_dataset('humans')
        dogs_files = DataLoader.read_dataset('dogs')

        self.assertTrue(len(human_files) == 13233)
        self.assertTrue(len(dogs_files) == 8351)

