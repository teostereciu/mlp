import numpy as np
from PIL import Image
from skimage import feature

from config.settings import RANDOM_SEED

import string
from zipfile import ZipFile
import os

import pandas as pd


class BasePreprocessor:
    def __init__(self, next_preprocessor=None):
        self.next_preprocessor = next_preprocessor

    def set_next_preprocessor(self, next_preprocessor):
        self.next_preprocessor = next_preprocessor
        return next_preprocessor

    def process(self, df):
        df = self.transform(df)
        if self.next_preprocessor:
            return self.next_preprocessor.process(df)
        else:
            return df

    def transform(self, df):
        # Default transformation: do nothing
        return df


class Sampler(BasePreprocessor):
    """
    Stratified sampling of the original dataset to obtain a mini-dataset.
    """

    def __init__(self, n=100):
        super().__init__()
        self.n = n  # how many samples to keep from each class

    def transform(self, df):
        print('sampler')
        return df.groupby('category', group_keys=False).apply(
            lambda x: x.sample(self.n, ignore_index=True, random_state=RANDOM_SEED))


def load_image(filename):
    try:
        image = Image.open(filename)
        grayscale_image = image.convert('L')
        image_array = np.array(grayscale_image)
        return image_array
    except Exception as e:
        print(f"Error loading image {filename}: {e}")
        return None


class GrayscaleLoader(BasePreprocessor):
    """
    Loads the images as grayscale numpy arrays.
    """

    def transform(self, df):
        print("grayscaler0")
        df['image_array'] = df['file_path'].apply(load_image)
        print("grayscaler")
        return df


class Normalizer(BasePreprocessor):
    """
    Normalize from 0-255 to 0-1.
    """

    def transform(self, df):
        df['image_array'] = df['image_array'] / 255
        print("normalizer")
        return df


class Flattener(BasePreprocessor):
    """
    Flatten 2d array to 1d.
    """

    def transform(self, df):
        df['image_array'] = df['image_array'].reshape(-1)
        return df


class CannyEdgesExtractor(BasePreprocessor):
    """
    Extract canny edges.
    """

    def transform(self, df):
        df['canny_edges'] = df['image_array'].apply(feature.canny)
        print("canny")
        return df


class DataHandler:
    # map alphabet to numbers
    categories = {i: letter for i, letter in enumerate(string.ascii_uppercase) if letter not in {'J', 'Z'}}

    # set up file paths
    current_directory = os.getcwd()
    os.path.dirname(current_directory)
    parent_directory = os.path.dirname(current_directory)
    dir_path = os.path.join(parent_directory, "data")
    train_path = os.path.join(dir_path, "asl_train")
    test_path = os.path.join(dir_path, "asl_test")

    df = None

    def unzip(self):
        self.unzip_if_not_exists(os.path.join(self.dir_path, "asl_alphabet_train.zip"), self.train_path)
        self.unzip_if_not_exists(os.path.join(self.dir_path, "asl_alphabet_test.zip"), self.test_path)

        # update file paths
        self.train_path += "/asl_alphabet_train/"
        self.test_path += "/asl_alphabet_test/"

    def unzip_if_not_exists(self, zip_file_path, extract_to_path):
        # check if the target directory already exists
        if not os.path.exists(extract_to_path):
            # create the directory if it doesn't exist
            os.makedirs(extract_to_path)
            # unzip the contents
            with ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to_path)
            print(f"Successfully unzipped to {extract_to_path}")
        else:
            print(f"The directory {extract_to_path} already exists. Skipped unzipping.")

    def create_filename_dataframe(self):
        def add_class_name_prefix(col_name):
            self.df[col_name]
            return self.df

        # store all the file names in the dataset
        filenames = []
        # store the corresponding class for each file
        target = []

        for category in self.categories:
            files = os.listdir(self.train_path + self.categories[category])
            filenames += files
            target += [category] * len(files)

        self.df = pd.DataFrame({"filename": filenames, "category": target})
        self.df = add_class_name_prefix("filename")
        self.df["file_path"] = self.df["filename"].apply(lambda x: os.path.join(self.train_path, x[0], x))

        return self.df
