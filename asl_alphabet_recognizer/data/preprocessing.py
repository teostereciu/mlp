import re

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from skimage import feature

from config.settings import RANDOM_SEED

import string
from zipfile import ZipFile
import os

import pandas as pd


def process_row(file_path):
    """
    Get images by filename, set them to grayscale,
    and normalize the pixel values.
    :param file_path: the path to each image
    :return: normalized_arr: the array representing the image
    """
    # load
    image = Image.open(file_path)
    # convert to grayscale
    gray_image = image.convert('L')
    # convert to 2d tensor
    image_arr = np.array(gray_image)
    # normalize to 0-1 range
    normalized_arr = image_arr / 255
    return normalized_arr


def unzip_if_not_exists(zip_file_path, extract_to_path):
    """
    Unzip a directory if it is not already unzipped.
    Used to unzip the original dataset.
    :param zip_file_path: where to find the zipped directory
    :param extract_to_path: where to save the unzipped directory
    """
    if not os.path.exists(extract_to_path):
        os.makedirs(extract_to_path)
        with ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to_path)
        print(f"Successfully unzipped to {extract_to_path}")
    else:
        print(f"The directory {extract_to_path} already exists. Skipped unzipping.")


def extract_float_arr(string_repr):
    values = re.findall(r'\d+\.\d+', string_repr)
    return np.array(values, dtype=float)


def extract_bool_arr(string_repr):
    values = re.findall(r'True|False', string_repr)
    bool_array = np.array(values == 'True', dtype=bool)
    num_columns = bool_array.size // bool_array.tolist().count(True) if bool_array.tolist().count(True) > 0 else 1
    return bool_array.reshape((-1, num_columns))

class DataHandler:
    """
    Handle the image dataset.
    """
    # map alphabet to numbers
    categories = {i: letter for i, letter in enumerate(string.ascii_uppercase) if letter not in {'J', 'Z'}}

    def __init__(self, data_dir):
        """
        Initialize data handler.
        :param data_dir: where to find the data to be handled
        """
        self.df = None
        self.n = 0
        self.df_train = None
        self.df_test = None
        self.data_dir = data_dir
        self.train_path = os.path.join(data_dir, "asl_train", "asl_alphabet_train")

    def unzip(self):
        """
        Unzip image data.
        :return: self
        """
        unzip_if_not_exists(os.path.join(self.data_dir, "asl_alphabet_train.zip"), self.train_path)
        return self

    def create_filename_dataframe(self):
        """
        Add image filenames and extract label to a dataframe.
        :return: self
        """
        filenames = []
        target = []

        for category in self.categories:
            files = os.listdir(os.path.join(self.train_path, self.categories[category]))
            filenames += files
            target += [category] * len(files)

        self.df = pd.DataFrame({"filename": filenames, "category": target})
        self.df["file_path"] = self.df["filename"].apply(lambda x: os.path.join(self.train_path, x[0], x))
        return self

    def sample(self, n=100, ratio=0.8):
        """
        Sample train and test sets.
        :param n: how many samples to use from the dataset for test and train per class
        :param ratio: train-test split
        :return: self
        """
        n_train = int(n * ratio)
        self.df = self.df.groupby('category', group_keys=False).apply(
            lambda x: x.sample(n, ignore_index=True, random_state=RANDOM_SEED))
        self.df_train = self.df.groupby('category', group_keys=False).apply(
            lambda x: x.sample(n_train, ignore_index=True, random_state=RANDOM_SEED))
        df_test = pd.merge(self.df, self.df_train, how='left', indicator=True)
        self.df_test = df_test[df_test['_merge'] == 'left_only'].drop(columns=['_merge']).reset_index(drop=True)

        return self

    def process(self):
        """
        Apply basic preprocessing to get number arrays for the images.
        :return: self
        """
        self.df_train['X'] = self.df_train['file_path'].apply(process_row)
        self.df_test['X'] = self.df_test['file_path'].apply(process_row)
        return self

    def apply_canny(self):
        """
        Apply Canny edge detection.
        :return: self
        """
        self.df_train['X_canny'] = self.df_train['X'].apply(feature.canny)
        self.df_test['X_canny'] = self.df_test['X'].apply(feature.canny)
        return self

    def get_dfs(self):
        """
        Return the train and test dataframes
        :return:
        """
        return self.df_train, self.df_test

    def save_dfs(self):
        """
        Save preprocessed train and test sets to memory as csv files.
        """
        train_path = os.path.join(self.data_dir, "train.csv")
        test_path = os.path.join(self.data_dir, "test.csv")
        self.df_train.to_csv(train_path, index=False, columns=['filename', 'category', 'X', 'X_canny'])
        self.df_test.to_csv(test_path, index=False, columns=['filename', 'category', 'X', 'X_canny'])

    def load_dfs(self, update_self=False):
        """
        Load preprocessed train and test sets from memory.
        :param update_self: whether to set the handler's train and test sets with the load
        :return df_train, df_test: the dataframes
        """
        train_path = os.path.join(self.data_dir, "train.csv")
        test_path = os.path.join(self.data_dir, "test.csv")
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
        df_train['X'] = df_train['X'].apply(extract_float_arr)
        df_test['X'] = df_test['X'].apply(extract_float_arr)
        df_train['X_canny'] = df_train['X_canny'].apply(extract_bool_arr)
        df_test['X_canny'] = df_test['X_canny'].apply(extract_bool_arr)
        if update_self:
            self.df_train = df_train
            self.df_test = df_test
        return df_train, df_test

    def show_before_and_after(self, idx=11):
        """
        Plot an example before and after processing image.
        :param idx:
        """
        plt.subplot(1, 2, 1)
        plt.imshow(self.df_train.iloc[idx]['X'], cmap='gray')
        plt.title('Grayscale image')

        plt.subplot(1, 2, 2)
        plt.imshow(self.df_train.iloc[idx]['X_canny'], cmap='gray')
        plt.title('Canny edges')

        plt.show()
