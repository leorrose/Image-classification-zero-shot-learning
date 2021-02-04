import os
import argparse
import zipfile
import cv2
import numpy as np
from typing import List
from urllib import request
from tensorflow.keras.models import load_model
from gensim.models import KeyedVectors
from gensim.models.keyedvectors import Word2VecKeyedVectors

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
    else:
        raise argparse.ArgumentTypeError(f"{path} is not a valid path")


def download_fast_text_vectors() -> None:
    """
    Download fast text vectors file data
    """
    url_download: str = "https://dl.fbaipublicfiles.com/fasttext/"
    url_download += "vectors-english/wiki-news-300d-1M.vec.zip"
    zip_name: str = 'wiki-news-300d-1M.vec.zip'
    file_name: str = "wiki-news-300d-1M.vec"

    # check if vectors exist already
    if not (os.path.exists(file_name) and os.path.isfile(file_name)):
        # download vectors zip
        with request.urlopen(url_download) as link_file:
            with open(zip_name, 'wb') as out_file:
                out_file.write(link_file.read())
        # extract vectors
        with zipfile.ZipFile("wiki-news-300d-1M.vec.zip", 'r') as zip_ref:
            zip_ref.extractall('.')

        # Delete vectors zip
        os.remove(zip_name)


def load_fast_text_vectors(file_name: str) -> Word2VecKeyedVectors:
    """
    Load fast text vectors file data
    Parameters:
        file_name (str): file name of word embedding vectors
    returns:
        Word2VecKeyedVectors: fast text vectors
    Raises:
        IOError: if file name doesn't exist
    """
    if os.path.isfile(file_name):
        return KeyedVectors.load_word2vec_format(
            file_name, binary=False, encoding='utf8')
    error: str = "Given fast text vectors file"
    error += f"name does not exist: {file_name}"
    raise IOError(error)


def predict_label(img: np.ndarray) -> str:
  # predict query
  prediction: np.ndarray = model.predict([img])
  # get top-n by cosine similarity
  most_similar: List[str] = fast_text_vectors.similar_by_vector(prediction[0], topn=5)
  return ", ".join([x[0] for x in most_similar])


if __name__ == '__main__':
    # create parser for command line arguments
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description='Image Classifier Arguments')
    parser.add_argument('image', type=file_path, help='the path to image')
    args = parser.parse_args()
    
    # download and load fasttext vectors
    download_fast_text_vectors()
    fast_text_vectors: Word2VecKeyedVectors = load_fast_text_vectors("wiki-news-300d-1M.vec")
    
    # load model h5
    model = load_model(f"{script_dir}/image_classification_model.h5")

    # load image and preprocess it
    image: np.ndarray = cv2.imread(args.image)
    image: np.ndarray = cv2.resize(image, (32, 32))
    image: np.ndarray = image.astype('float32')
    image /= 255.0
    image: np.ndarray = np.expand_dims(image, axis=0)

    # predict query
    prediction: str = predict_label(image)
    print(f"Prediction for image: {prediction}")
