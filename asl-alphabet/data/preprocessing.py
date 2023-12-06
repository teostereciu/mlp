import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from skimage import feature

from config.settings import RANDOM_SEED

import string
from zipfile import ZipFile
import os

import pandas as pd


def unzip_if_not_exists(zip_file_path, extract_to_path):
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


def process_row(file_path):
    # load
    image = Image.open(file_path)
    # convert to grayscale
    gray_image = image.convert('L')
    # convert to 2d tensor
    image_arr = np.array(gray_image)
    # normalize to 0-1 range
    normalized_arr = image_arr / 255
    # flatten 2d tensor to 1d array
    # flat_image = normalized_arr.ravel()
    return normalized_arr


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
    df_processed = None

    def unzip(self):
        unzip_if_not_exists(os.path.join(self.dir_path, "asl_alphabet_train.zip"), self.train_path)
        unzip_if_not_exists(os.path.join(self.dir_path, "asl_alphabet_test.zip"), self.test_path)

        # update file paths
        self.train_path += "/asl_alphabet_train/"
        self.test_path += "/asl_alphabet_test/"
        return self

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
        return self

    def sample(self, n=100):
        self.df_processed = self.df.groupby('category', group_keys=False).apply(
            lambda x: x.sample(n, ignore_index=True, random_state=RANDOM_SEED))
        return self

    def process(self):
        self.df_processed['X'] = self.df_processed['file_path'].apply(process_row)
        return self

    def apply_canny(self):
        self.df_processed['X_canny'] = self.df_processed['X'].apply(feature.canny)
        return self

    def show_before_and_after(self, idx=11):
        plt.subplot(1, 2, 1)
        plt.imshow(self.df_processed.iloc[idx]['X'], cmap='gray')
        plt.title('Grayscale image')

        plt.subplot(1, 2, 2)
        plt.imshow(self.df_processed.iloc[idx]['X_canny'], cmap='gray')
        plt.title('Canny edges')

        plt.show()

    def get_original_df(self):
        return self.df

    def get_preprocessed_df(self):
        return self.df_processed
