from data.preprocessing import DataHandler
from config.settings import data_dir
import argparse


def main():
    # parser = argparse.ArgumentParser(description='ASL alphabet recognizer project.')
    # parser.add_argument('--skip_preprocess', type=bool, default=False, help='Whether to skip preprocessing')
    # parser.add_argument('--skip_training', type=str, default=False, help='Whether to skip training')

    # args = parser.parse_args()

    data_handler = DataHandler(data_dir)
    (data_handler.unzip()
     .create_filename_dataframe()
     .sample()
     .process()
     .apply_canny())

    data_handler.show_before_and_after()

    df_train, df_test = data_handler.get_dfs()

    print(df_train.head())

if __name__ == '__main__':
    main()
