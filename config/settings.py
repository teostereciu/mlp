import os

RANDOM_SEED = 42
current_directory = os.getcwd()
parent_directory = os.path.dirname(current_directory)
data_dir = os.path.join(parent_directory, "data")