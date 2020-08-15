import argparse, os, sys
import gensim.models.wrappers.fasttext
import tensorflow.keras.datasets.cifar100 as ds
import numpy as np
import urllib.request
import zipfile
from cv2 import cv2
from keras.optimizers import Adam
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Activation, Flatten
from keras.preprocessing.text import text_to_word_sequence

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

def parse_cmd():
    """
    Parse cmd launch program to get all arguments.
    For test/train arguments are different.
    """
    try:
        parser = argparse.ArgumentParser(description='Input format')
        parser.add_argument('--task', action="store", type=str, required=True, help='<type of task - train/test>', choices=['train', 'test'])
        parser.add_argument('--image', action="store", type=str, required=False, help='<image file name>')
        parser.add_argument('--model', action="store", type=str, required=True, help='<trained model>')
        args = parser.parse_args()

        # Validate model is .h5 file
        _, model_file_extension = os.path.splitext(args.model)
        if model_file_extension != '.h5':
            raise ValueError("--model needs to be a .h5 file")

        if args.task == "test":
            # check image variable was given
            if not args.image:
                raise ValueError("--image is required for testing")

            # check image file was given
            if not (os.path.exists(args.image) and os.path.isfile(args.image)):
                raise IOError("--image file doesn't exist")

            # check model exist
            if not (os.path.exists(args.model) and os.path.isfile(args.model)):
                raise IOError("--model file doesn't exist")

    except (ValueError, TypeError, IOError) as exp:
        print("An Error Accrued When trying to parse program arguments: {0}".format(exp))
        sys.exit()

    return args

def download_fast_text_vectors():
    """
    Download fast text vectors file data
    """
    url_download = "https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip"
    zip_name = 'wiki-news-300d-1M.vec.zip'
    file_name = "wiki-news-300d-1M.vec"

    # check if vectors exist already
    if not (os.path.exists(file_name) and os.path.isfile(file_name)):
        # download vectors zip
        with urllib.request.urlopen(url_download) as link_file:
            with open(zip_name, 'wb') as out_file:
                out_file.write(link_file.read())
        # extract vectors
        with zipfile.ZipFile("wiki-news-300d-1M.vec.zip", 'r') as zip_ref:
            zip_ref.extractall('.')

        # Delete vectors zip
        os.remove(zip_name)

def load_fast_text_vectors(file_name):
    """
    Load fast text vectors file data
    Parameters:
        file_name (str): file name of word embedding vectors
    returns:
        gensim.models.keyedvectors.Word2VecKeyedVectors: fast text vectors
    Raises:
        IOError: if file name doesn't exist
    """
    if os.path.isfile(file_name):
        return gensim.models.KeyedVectors.load_word2vec_format(file_name, binary=False, encoding='utf8')
    raise IOError("Given fast text vectors file name does not exist: {0}".format(file_name))

def word_list_to_fast_text_vector(lst, fast_text_vectors):
    """
    Transform a string to fast text vector
    Parameters:
        sentence (str): string to get fast text vector
        fast_text_vectors (gensim.models.keyedvectors.Word2VecKeyedVectors): fast text vectors
    Returns:
        numpy.array: fast text representation of sentence
    """
    # create empty array
    sum_of_vec = np.array([0.0] * 300)

    # run over each word
    for word_val in lst:
        # add values to word to sentence representation
        sum_of_vec += fast_text_vector_for_word(word_val, fast_text_vectors)

    # get average fast text vector
    sum_of_vec /= len(lst)
    sum_of_vec[np.isnan(sum_of_vec)] = 0
    return sum_of_vec

def fast_text_vector_for_word(word, fast_text_vectors):
    """
    Transform a string word to fast text vector
    Parameters:
        word (str): word to get fast text vector
        fast_text_vectors (gensim.models.keyedvectors.Word2VecKeyedVectors): fast text vectors
    Returns:
        numpy.array: fast text representation of word
    """
    try:
        return fast_text_vectors.word_vec(word)
    except KeyError:
        return np.zeros(300)

def create_model(fast_text_vectors):
    """
    Function to image classification model
    Parameters:
        fast_text_vectors (gensim.models.keyedvectors.Word2VecKeyedVectors): fast text vectors
    """
    # load images
    (x_train, y_train), (x_test, y_test) = ds.load_data(label_mode="fine")

    # images class each index is the position of class => image_class = 0 then classes[0] = apple
    classes = [
        'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
        'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
        'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
        'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
        'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
        'house', 'kangaroo', 'computer_keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
        'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
        'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
        'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
        'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
        'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
        'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
        'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
        'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
        'worm'
    ]
    
    # mapper of each class to its super class
    mapping = {
        'apple': 'fruit vegetables', 'aquarium_fish': 'fish', 'baby': 'people', 'bear': 'large carnivores',
        'beaver': 'aquatic mammals', 'bed':'household furniture', 'bee': 'insects', 'beetle': 'insects',
        'bicycle': 'vehicles 1', 'bottle': 'food containers', 'bowl': 'food containers', 'boy': 'people',
        'bridge': 'large man-made outdoor things', 'bus': 'vehicles 1', 'butterfly': 'insects', 'camel': 'large omnivores herbivores',
        'can': 'food containers', 'castle': 'large man-made outdoor things', 'caterpillar': 'insects',
        'cattle': 'large omnivores herbivores', 'chair': 'household furniture', 'chimpanzee': 'large omnivores herbivores',
        'clock': 'household electrical device', 'cloud': 'large natural outdoor scenes', 'cockroach': 'insects', 'couch': 'household furniture',
        'crab': 'non-insect invertebrates', 'crocodile': 'reptiles', 'cup': 'food containers',
        'dinosaur': 'reptiles', 'dolphin': 'aquatic mammals', 'elephant': 'large omnivores herbivores',
        'flatfish': 'fish', 'forest': 'large natural outdoor scenes', 'fox': 'medium-sized mammals', 'girl': 'people',
        'hamster': 'small mammals', 'house': 'large man-made outdoor things', 'kangaroo': 'large omnivores herbivores',
        'computer_keyboard': 'household electrical device', 'lamp': 'household electrical device', 'lawn_mower': 'vehicles 2',
        'leopard': 'large carnivores', 'lion': 'large carnivores', 'lizard': 'reptiles',
        'lobster': 'non-insect invertebrates', 'man': 'people', 'maple_tree': 'trees', 'motorcycle': 'vehicles 1',
        'mountain': 'large natural outdoor scenes', 'mouse': 'small mammals', 'mushroom': 'fruit vegetables',
        'oak_tree': 'trees', 'orange': 'fruit vegetables', 'orchid': 'flowers', 'otter': 'aquatic mammals',
        'palm_tree': 'trees', 'pear': 'fruit vegetables', 'pickup_truck': 'vehicles 1', 'pine_tree': 'trees',
        'plain': 'large natural outdoor scenes', 'plate': 'food containers', 'poppy': 'flowers', 'porcupine': 'medium-sized mammals' ,
        'possum': 'medium-sized mammals', 'rabbit': 'small mammals', 'raccoon': 'medium-sized mammals', 'ray': 'fish',
        'road': 'large man-made outdoor things', 'rocket': 'vehicles 2', 'rose': 'flowers',
        'sea': 'large natural outdoor scenes', 'seal': 'aquatic mammals', 'shark': 'fish', 'shrew': 'small mammals', 
        'skunk': 'medium-sized mammals', 'skyscraper': 'large man-made outdoor things', 'snail': 'non-insect invertebrates',
        'snake': 'reptiles', 'spider': 'non-insect invertebrates', 'squirrel': 'small mammals', 'streetcar': 'vehicles 2',
        'sunflower': 'flowers', 'sweet_pepper': 'fruit vegetables', 'table': 'household furniture',
        'tank':'vehicles 2', 'telephone': 'household electrical device', 'television': 'household electrical device',
        'tiger': 'large carnivores', 'tractor': 'vehicles 2', 'train': 'vehicles 1', 'trout': 'fish',
        'tulip': 'flowers', 'turtle': 'reptiles', 'wardrobe': 'household furniture', 'whale': 'aquatic mammals',
        'willow_tree': 'trees', 'wolf': 'large carnivores', 'woman': 'people', 'worm':'non-insect invertebrates'
    }

    # get fast text vectors for each class
    regular_classes_fast_text_vector = [word_list_to_fast_text_vector(text_to_word_sequence(class_val), fast_text_vectors) for class_val in classes]
    
    # get fast text vectors for each super class of class
    super_classes_fast_text_vector = [word_list_to_fast_text_vector(text_to_word_sequence(mapping[class_val]), fast_text_vectors) for class_val in classes]
    
    # create fast text for each class considering super class and class
    classes_fast_text_vector = np.asarray(regular_classes_fast_text_vector) * 0.75 + (np.asarray(super_classes_fast_text_vector)/2) * 0.25
    
    # transform labels to fast text vectors
    y_train = np.asarray([classes_fast_text_vector[class_integer[0]] for class_integer in y_train])
    y_test = np.asarray([classes_fast_text_vector[class_integer[0]] for class_integer in y_test])
    
    # normalize images
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255.0
    x_test /= 255.0

    # Params for model
    input_shape = (32, 32, 3)
    chan_dim = -1

    #Create model
    model = Sequential()
    
    # Add CNN layers
    #  Stack 1:
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=chan_dim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Stack 2:
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=chan_dim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Stack 3:
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=chan_dim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Stack 4:
    model.add(Flatten())
    model.add(Dense(300))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # output layer:
    model.add(Dense(300))

    # Compile model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['accuracy'])

    # train
    model.fit(x_train, y_train, epochs=20, shuffle=True, batch_size=32)

    # eval model
    model_eval = model.evaluate(x_test, y_test)
    print("model 3 loss {0} model 3 accuracy {1}".format(model_eval[0], model_eval[1]))

    return model

if __name__ == '__main__':
    try:
        # parse cmd arguments
        args = parse_cmd()
        
        # download fast text vectors 
        download_fast_text_vectors()
        # load fast text vectors
        fast_text_vectors = load_fast_text_vectors("wiki-news-300d-1M.vec")

        if args.task == "train":
            model = create_model(fast_text_vectors)
            model.save(args.model)

        elif args.task == "test":
            # load model h5
            model = load_model(args.model)

            # load image and preprocess it
            image = cv2.imread(args.image)
            image = cv2.resize(image, (32, 32))
            image = image.astype('float32')
            image /= 255.0
            image = np.expand_dims(image, axis=0)

            # predict query
            prediction = model.predict(image)

            # get all cosine similarity of prediction and fast text words
            most_similar = fast_text_vectors.similar_by_vector(prediction[0], topn=5)

            # print results
            print(", ".join([x[0] for x in most_similar]))

    # if any exception accours in run
    except Exception as exp:
        #print("An Error: {0}".format(exp))
        raise exp    
