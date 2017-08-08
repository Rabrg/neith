from os import listdir
from os.path import isfile, join
from PIL import Image
import numpy as np

DATASET_PATH = '../res/'
CLASS_INDEX = ['(', ')', '+', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'slash', 'star']
NUM_CLASSES = len(CLASS_INDEX)
IMG_ROWS = 32
IMG_COLS = 32


def load_dataset():
    files = [f for f in listdir(DATASET_PATH) if isfile(join(DATASET_PATH, f))]
    features = np.empty((len(files), IMG_ROWS, IMG_COLS))
    labels = []
    for i, file in enumerate(files):
        feature = np.array(Image.open(DATASET_PATH + file).convert(mode='L'))
        features[i] = feature
        labels.append(CLASS_INDEX.index(str(file.split("_")[0])))
    labels = np.asarray(labels)
    np.divide(features, 255, out=features)
    np.subtract(features, 0.5, out=features)
    return features, labels
