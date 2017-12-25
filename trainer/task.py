import pandas as pd
import tensorflow as tf
import numpy as np
import csv
import logging

import argparse

from sklearn.model_selection import train_test_split

from tensorflow.python.lib.io import file_io

class model():
    def __init__(self):
        mean = 0
        std = 0.1
        self.weights = {'wc1': tf.Variable(tf.truncated_normal((5, 5, 1, 32), mean=mean, stddev=std, dtype=tf.float32),
                                      name='ConvolutionalWeight1'),
                   'wc2': tf.Variable(tf.truncated_normal((5, 5, 32, 64), mean=mean, stddev=std, dtype=tf.float32),
                                      name='ConvolutionalWeight2'),
                   'wfc1': tf.Variable(tf.truncated_normal((3136, 1024), mean=mean, stddev=std, dtype=tf.float32),
                                       name='FullyConnectedLayerWeight1'),
                   'wfc2': tf.Variable(tf.truncated_normal((1024, 10), mean=mean, stddev=std, dtype=tf.float32),
                                       name='FullyConnectedLayerWeight2'),
                   }

        self.biases = {'bc1': tf.zeros(32, name='ConvolutionalBias1'),
                  'bc2': tf.zeros(64, name='ConvolutionalBias2'),
                  'bfc1': tf.zeros(1024, name='FullyConnectedLayerBias1'),
                  'bfc2': tf.zeros(10, name='FullyConnectedLayerBias2'),
                  }

        self.X = tf.placeholder(tf.float32, (None, 28, 28, 1), name='input')
        self.Y = tf.placeholder(tf.int32, (None), name='label')
        self.onehot_Y = tf.one_hot(self.Y, 10)


        def conv2d(x, W):
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

        def max_pool_2(x):
            return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # Layer 1: Convolution Layer
        self.conv1 = tf.nn.relu(conv2d(self.X, self.weights['wc1']) + self.biases['bc1'])
        self.max_pool1 = max_pool_2(self.conv1)

        # Layer 2: Convolution Layer
        self.conv2 = tf.nn.relu(conv2d(self.max_pool1, self.weights['wc2']) + self.biases['bc2'])
        self.max_pool2 = max_pool_2(self.conv2)

        # Layer 3: Fully Connected
        self.fc1_input = tf.reshape(self.max_pool2, [-1, 3136])
        self.fc1 = tf.nn.relu(tf.matmul(self.fc1_input, self.weights['wfc1']) + self.biases['bfc1'])

        # Layer 4: Dropout
        self.keep_prob = tf.placeholder(tf.float32)
        self.dropout = tf.nn.dropout(self.fc1, self.keep_prob)

        # Layer 5: Fully Connected
        self.output = tf.matmul(self.dropout, self.weights['wfc2']) + self.biases['bfc2']

        # For Training
        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.onehot_Y, logits=self.output))

        # Using ADAM optimizer to train the network
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)
        self.correct_prediction = tf.equal(tf.argmax(self.output, 1), tf.argmax(self.onehot_Y, 1))

    # Function to convert one hot output to label
    def onehot2Label(self, soft_output):
        label = []
        for a in soft_output:
            label.append(np.argmax(a))
        return label

    # Training Function
    def fit(self, X_train, X_test, y_train, y_test, job_dir, batch_size=100, epochs=20000):
        # For Training
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.onehot_Y, logits=self.output))

        # Using ADAM optimizer to train the network
        train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)
        correct_prediction = tf.equal(tf.argmax(self.output, 1), tf.argmax(self.onehot_Y, 1))
        accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            logging.info("Training Begins")
            sess.run(tf.global_variables_initializer())

            accuracy_log = []
            for i in range(epochs):
                num_examples = len(X_test)


                for offset in range(0, batch_size):
                    end = offset + batch_size
                    batch_x, batch_y = X_train[offset:end], y_train[offset:end]
                    sess.run(train_step, feed_dict={self.X: batch_x, self.Y: batch_y, self.keep_prob: 0.5})

                if (i + 1) % 200 == 0:

                    total_accuracy = 0

                    for offset in range(0, batch_size):
                        batch_vx, batch_vy = X_test[offset:end], y_test[offset:end]
                        accu = sess.run(accuracy, feed_dict={self.X: batch_vx, self.Y: batch_vy, self.keep_prob: 1.0})
                        total_accuracy += (accu * len(batch_vx))

                    validation_accuracy = total_accuracy / num_examples
                    accuracy_log.append([validation_accuracy])
                    logging.info("Epoch " + str(i + 1) + " Validation Accuracy: " + str(validation_accuracy))

            with file_io.FileIO(job_dir + "/accuracy.csv", 'w') as f:
                writer = csv.writer(f)
                writer.writerows(accuracy_log)

            save_path = tf.train.Saver().save(sess, save_path= job_dir + "/model1.ckpt")

    # Testing Function
    def predict(self, features, job_dir):
        test_softmax = tf.nn.softmax(self.output)

        test_batch_size = 1000

        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, job_dir + "/model1.ckpt")

            all_labels = []

            i = 0

            for start in range(0, features.shape[0], test_batch_size):
                end = start + test_batch_size
                batch_test = features[start:end]
                result = sess.run(self.output, feed_dict={self.X: batch_test, self.keep_prob: 1.0})
                result_soft = sess.run(test_softmax, feed_dict={self.output: result})

                test_labels = self.onehot2Label(result_soft)

                all_labels.append(test_labels)
                i += 1

        return np.reshape(all_labels, (len(features)))



def getFeatures(data):
    features = []
    for row in data:
        tmp_img = np.zeros((28, 28))
        for i in range(0, 28):
            for j in range(0, 28):
                tmp_img[i][j] = row[i * 28 + j]
        tmp_img = np.reshape(tmp_img, (28, 28, 1))
        features.append(tmp_img)

    # Printing Features Matrix Shape
    return np.array(features)


def main(_):
    logging.info("Program Started")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-dir',
        help='Cloud Storage bucket or local path to training data')
    parser.add_argument(
        '--job-dir',
        help='Cloud storage bucket to export the model and store temp files')


    args = parser.parse_args()

    arguments = args.__dict__

    job_dir = arguments['job_dir']
    data_dir = arguments['data_dir']

    with file_io.FileIO(data_dir + "/train.csv", 'r') as f:
        data = pd.read_csv(f)

    logging.info("Training Data Read")
    labels = data['label'].values
    features = getFeatures(data.drop(['label'], axis=1).values)

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.33)

    digit_model = model()

    digit_model.fit(X_train, X_test, y_train, y_test, job_dir=job_dir)

    with file_io.FileIO(data_dir + "/test.csv", 'r') as f:
        test_data = pd.read_csv(f)

    # Dividing Pandas Data Into Labels and Features
    test_features = getFeatures(test_data.values)

    test_labels = digit_model.predict(test_features, job_dir)

    img_id = np.arange(1, len(test_features) + 1)
    data = np.transpose(np.vstack((img_id, test_labels)))

    with file_io.FileIO(job_dir + "/result-digit.csv", 'w') as f:
        writer = csv.writer(f)

        writer.writerows([['ImageId', 'Label']])
        writer.writerows(data)


if __name__ == '__main__':
    tf.app.run()