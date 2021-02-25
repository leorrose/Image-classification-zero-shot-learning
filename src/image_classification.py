import os
import argparse
import numpy as np
import tensorflow as tf
import gensim as gs
import gensim.downloader as gdownloader
from typing import List

script_dir: str = os.path.dirname(os.path.realpath(__file__))


def file_path(path: str) -> str:
    """
    Method to test if path is a file

    Args:
        path (str): the path
    Raises:
        argparse.ArgumentTypeError: if path is not a file

    Returns:
        str: the path if it is a file
    """
    if os.path.isfile(path):
        return path
    raise argparse.ArgumentTypeError(f"{path} is not a valid path")


if __name__ == '__main__':
    # create parser for command line arguments
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description='Image Classifier Arguments')
    parser.add_argument('image', type=file_path, help='the path to image')
    img_path: str = parser.parse_args().image

    # download and load fasttext vectors
    fast_text_vectors: gs.models.keyedvectors = gdownloader.load(
        "fasttext-wiki-news-subwords-300")

    # load model h5
    model: tf.keras.Model = tf.keras.models.load_model(
        f"{script_dir}/model.h5")

    # load img
    img: np.ndarray = np.asarray(
        tf.keras.preprocessing.image.load_img(img_path))

    # get prediction vector
    prediction: np.ndarray = model.predict(np.expand_dims(
        tf.keras.applications.vgg19.preprocess_input(
            tf.image.resize(img, (32, 32))), axis=0))

    # get top-n labels by cosine similarity
    most_similar: List[str] = fast_text_vectors.similar_by_vector(
        prediction[0], topn=5)

    # print the predictions for image
    print(f"Prediction for image: {', '.join([x[0] for x in most_similar])}")
