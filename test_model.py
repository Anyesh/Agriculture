from keras.models import load_model
from keras.utils import np_utils, plot_model
from sklearn.preprocessing import LabelEncoder
import pickle
import pandas as pd
import numpy as np
import argparse
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error(f"The file {arg} does not exist!")
    else:
        return arg



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', type=lambda x: is_valid_file(parser, x), default='data_labels.dat', help='enter the label filename', metavar='FILE')
    parser.add_argument('--model', type=lambda x: is_valid_file(parser, x), default='data_model.hdf5', help='enter the model filename', metavar='FILE')
    parser.add_argument('--test', type=lambda x: is_valid_file(parser, x), help='enter the test file name', metavar='FILE')

    args = parser.parse_args()
    sys.stdout.write(str(test_model(args)))


def test_model(args):

    label = args.label
    with open(label, 'rb') as f:
        lb = pickle.load(f)

    model = load_model(args.model)
    test = [
        0.242, 0, 0.124, 0.453, 0.053, 0.18, 0.428, 0.184, 0.272, 0.463, 0.693,
        0.594
    ]
    test = np.array(test).reshape((1, 12))
    prediction = model.predict_proba(test, verbose=1)
    top_k = prediction[0].argsort()[-len(prediction[0]):][::-1][:3]

    for node_id in top_k:
        test_label = lb.inverse_transform(node_id)
        score = prediction[0][node_id]
        print('%s (score = %.5f)' % (test_label, score))

if __name__ == '__main__':
    main()
