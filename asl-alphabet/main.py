from data.preprocessing import DataHandler


def main():
    data_handler = DataHandler()
    (data_handler.unzip()
     .create_filename_dataframe()
     .sample()
     .process()
     .apply_canny()
     .show_before_and_after())


if __name__ == '__main__':
    main()
