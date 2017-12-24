import numpy as np
import pandas as pd
import pickle
import timeit
from time import time
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.callbacks import TensorBoard
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import argparse
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return open(arg, 'r')  # return an open file handle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=lambda x: is_valid_file(parser, x), default='new_data1.csv', help='Enter the dataset.', metavar='FILE')
    parser.add_argument(
        '--l', type=float, default=0.01, help='Enter the learning rate')
    parser.add_argument(
        '--batch', type=int, default=50, help='Enter the batch size')
    parser.add_argument(
        '--epoch', type=int, default=100, help='Enter the batch epochs')
    parser.add_argument(
        '--input', type=int, default=12, help='enter the input dimension')
    parser.add_argument(
        '--output', type=int, default=36, help='enter the output size')

    args = parser.parse_args()
    sys.stdout.write(str(train_model(args)))


def train_model(args):

    dataset = args.dataset
    df = pd.read_csv(dataset)

    label = df['Label']
    label = label.as_matrix()

    # df.drop(df.columns[0], axis=1, inplace=True)
    del df['Label']
    data = df.as_matrix()

    lb = LabelEncoder().fit(label)
    encoded_Y = lb.transform(label)
    dummy_y = np_utils.to_categorical(encoded_Y)
    (x_train, x_test, y_train, y_test) = train_test_split(
        data, dummy_y, test_size=0.25, random_state=0, stratify=label)

    with open('data_labels.dat', "wb") as f:
        pickle.dump(lb, f)

    # shape of x (360036, 12) y (360036, 36)
    print(x_train.shape, y_train.shape)
    # shape of x (270027, 12) y (270027, 36)
    print(x_test.shape, y_test.shape)

    # shape of x (90009, 12) y (90009, 36)
    # 12 dimension of data and 36 output

    model = Sequential()

    model.add(Dense(64, activation='relu', input_dim=args.input))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(args.output, activation='softmax'))

    # sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(
        loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
    model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=args.epoch,
        batch_size=args.batch,
        verbose=1,
        callbacks=[tensorboard])
    model.save('data_model.hdf5')
    score = model.evaluate(x_test, y_test, batch_size=args.batch)
    print("Accuracy Score:", score)

if __name__ == '__main__':
    try:
        main()
    except ValueError:
        print('[!] Enter the correct params [!]')
