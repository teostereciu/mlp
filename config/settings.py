import os

RANDOM_SEED = 21
current_directory = os.getcwd()
parent_directory = os.path.dirname(current_directory)
grandparent_directory = os.path.dirname(parent_directory)
data_dir = os.path.join(grandparent_directory, "data")
debug = True
