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
    if not os.path.exists(extract_to_path):
        os.makedirs(extract_to_path)
        with ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to_path)
        print(f"Successfully unzipped to {extract_to_path}")
    else:
        print(f"The directory {extract_to_path} already exists. Skipped unzipping.")


class DataHandler:
    # map alphabet to numbers
    categories = {i: letter for i, letter in enumerate(string.ascii_uppercase) if letter not in {'J', 'Z'}}
    df = None
    n = 0
    df_train = None
    df_test = None

    def __init__(self, data_dir):

        self.data_dir = data_dir
        self.train_path = os.path.join(data_dir, "asl_train", "asl_alphabet_train")
        self.test_path = os.path.join(data_dir, "asl_test", "asl_alphabet_test")

    def unzip(self):
        unzip_if_not_exists(os.path.join(self.data_dir, "asl_alphabet_train.zip"), self.train_path)
        unzip_if_not_exists(os.path.join(self.data_dir, "asl_alphabet_test.zip"), self.test_path)
        return self

    def create_filename_dataframe(self):
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
        n_train = int(n * ratio)
        self.df = self.df.groupby('category', group_keys=False).apply(
            lambda x: x.sample(n, ignore_index=True, random_state=RANDOM_SEED))
        self.df_train = self.df.groupby('category', group_keys=False).apply(
            lambda x: x.sample(n_train, ignore_index=True, random_state=RANDOM_SEED))
        df_test = pd.merge(self.df, self.df_train, how='left', indicator=True)
        self.df_test = df_test[df_test['_merge'] == 'left_only'].drop(columns=['_merge']).reset_index(drop=True)

        return self

    def process(self):
        self.df_train['X'] = self.df_train['file_path'].apply(process_row)
        self.df_test['X'] = self.df_test['file_path'].apply(process_row)
        return self

    def apply_canny(self):
        self.df_train['X_canny'] = self.df_train['X'].apply(feature.canny)
        self.df_test['X_canny'] = self.df_test['X'].apply(feature.canny)
        return self

    def get_dfs(self):
        return self.df_train, self.df_test

    def show_before_and_after(self, idx=11):
        plt.subplot(1, 2, 1)
        plt.imshow(self.df_train.iloc[idx]['X'], cmap='gray')
        plt.title('Grayscale image')

        plt.subplot(1, 2, 2)
        plt.imshow(self.df_train.iloc[idx]['X_canny'], cmap='gray')
        plt.title('Canny edges')

        plt.show()
