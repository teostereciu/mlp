from data.preprocessing import DataHandler, Sampler, GrayscaleLoader, Normalizer, CannyEdgesExtractor


def main():
    data_handler = DataHandler()
    data_handler.unzip()
    df = data_handler.create_filename_dataframe()
    print(df.head())
    sampler = Sampler(n=1)
    grayscale_loader = GrayscaleLoader()
    normalizer = Normalizer()
    canny = CannyEdgesExtractor()

    preprocessor_chain = sampler.set_next_preprocessor(grayscale_loader)

    processed_df = preprocessor_chain.process(df)
    print(df.head())

if __name__ == '__main__':
    main()
