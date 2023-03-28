# -*- coding:utf-8 -*-
# @FileName :svm.py
# @Time :2023/3/24 13:34
# @Author :Xiaofeng
import random
import numpy as np
from tqdm import tqdm

from cs231n.data_utils import load_CIFAR10
import matplotlib.pyplot as plt
from cs231n.classifiers.linear_svm import svm_loss_naive
from cs231n.classifiers import LinearSVM
from cs231n.classifiers.linear_svm import svm_loss_vectorized
import time


def preprocess_data(num_training=49000, num_validation=1000, num_test=1000, num_dev=500):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the linear classifier. These are the same steps as we used for the
    SVM, but condensed to a single function.
    """
    # Load the raw CIFAR-10 data.
    cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
    # # Cleaning up variables to prevent loading data multiple times (which may cause memory issue)
    # try:
    #     del X_train, y_train
    #     del X_test, y_test
    #     print('Clear previously loaded data.')
    # except:
    #     pass

    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # subsample the data
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]
    mask = np.random.choice(num_training, num_dev, replace=False)
    X_dev = X_train[mask]
    y_dev = y_train[mask]

    # Preprocessing: reshape the image data into rows
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_val = np.reshape(X_val, (X_val.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image
    X_dev -= mean_image

    # add bias dimension and transform into columns
    X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
    X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])

    return X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev


def runSVM():
    X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev = preprocess_data()
    # generate a random SVM weight matrix of small numbers
    W = np.random.randn(3073, 10) * 0.0001

    loss, grad = svm_loss_naive(W, X_dev, y_dev, 0.000005)
    print('loss: %f' % (loss,))
    # Next implement the function svm_loss_vectorized; for now only compute the loss;
    # we will implement the gradient in a moment.
    tic = time.time()
    loss_naive, grad_naive = svm_loss_naive(W, X_dev, y_dev, 0.000005)
    toc = time.time()
    print('Naive loss: %e computed in %fs' % (loss_naive, toc - tic))

    tic = time.time()
    loss_vectorized, _ = svm_loss_vectorized(W, X_dev, y_dev, 0.000005)
    toc = time.time()
    print('Vectorized loss: %e computed in %fs' % (loss_vectorized, toc - tic))

    # The losses should match but your vectorized implementation should be much faster.
    print('difference: %f' % (loss_naive - loss_vectorized))


def run_LinearSVM():
    # X:原始的图片数据集(N个实例,D维),y:对应的所属的类别,即标签
    X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev = preprocess_data()
    svm = LinearSVM()
    tic = time.time()
    loss_hist = svm.train(X_train, y_train, learning_rate=1e-7, reg=2.5e4,
                          num_iters=1500, verbose=True)
    toc = time.time()
    print('That took %fs' % (toc - tic))
    # A useful debugging strategy is to plot the loss as a function of
    # iteration number:
    plt.plot(loss_hist)
    plt.xlabel('Iteration number')
    plt.ylabel('Loss value')
    plt.show()

    # Write the LinearSVM.predict function and evaluate the performance on both the
    # training and validation set
    y_train_pred = svm.predict(X_train)
    print('training accuracy: %f' % (np.mean(y_train == y_train_pred),))
    y_val_pred = svm.predict(X_val)
    print('validation accuracy: %f' % (np.mean(y_val == y_val_pred),))


def tune_hyperparameters():
    X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev = preprocess_data()
    svm = LinearSVM()
    tic = time.time()
    # Use the validation set to tune hyperparameters (regularization strength and
    # learning rate). You should experiment with different ranges for the learning
    # rates and regularization strengths; if you are careful you should be able to
    # get a classification accuracy of about 0.39 (> 0.385) on the validation set.

    # Note: you may see runtime/overflow warnings during hyper-parameter search.
    # This may be caused by extreme values, and is not a bug.

    # results is dictionary mapping tuples of the form
    # (learning_rate, regularization_strength) to tuples of the form
    # (training_accuracy, validation_accuracy). The accuracy is simply the fraction
    # of data points that are correctly classified.
    results = {}
    best_val = -1  # The highest validation accuracy that we have seen so far.
    best_svm = None  # The LinearSVM object that achieved the highest validation rate.

    ################################################################################
    # TODO:                                                                        #
    # Write code that chooses the best hyperparameters by tuning on the validation #
    # set. For each combination of hyperparameters, train a linear SVM on the      #
    # training set, compute its accuracy on the training and validation sets, and  #
    # store these numbers in the results dictionary. In addition, store the best   #
    # validation accuracy in best_val and the LinearSVM object that achieves this  #
    # accuracy in best_svm.                                                        #
    #                                                                              #
    # Hint: You should use a small value for num_iters as you develop your         #
    # validation code so that the SVMs don't take much time to train; once you are #
    # confident that your validation code works, you should rerun the validation   #
    # code with a larger value for num_iters.                                      #
    ################################################################################

    # Provided as a reference. You may or may not want to change these hyperparameters
    learning_rates = [2e-8, 1.5e-7]
    regularization_strengths = [2.5e4, 5e4]
    # lr 7.333333e-08 reg 2.500000e+04 train accuracy: 0.380796 val accuracy: 0.409000
    learning_rates_range = np.linspace(learning_rates[0], learning_rates[1], 6)
    regularization_strengths_range = np.linspace(regularization_strengths[0], regularization_strengths[1], 5)
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    for lr in tqdm(learning_rates_range):
        for reg in regularization_strengths_range:
            loss_hist = svm.train(X_train, y_train, learning_rate=lr, reg=reg, num_iters=1500, verbose=False)
            y_train_pred = svm.predict(X_train)
            acc_train = np.mean(y_train == y_train_pred)
            y_val_pred = svm.predict(X_val)
            acc_val = np.mean(y_val == y_val_pred)
            results[(lr, reg)] = (acc_train, acc_val)
            if acc_val > best_val:
                best_val = acc_val
                best_svm = svm
            print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
                lr, reg, acc_train, acc_val))

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Print out results.
    # for lr, reg in sorted(results):
    #     train_accuracy, val_accuracy = results[(lr, reg)]
    #     print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
    #         lr, reg, train_accuracy, val_accuracy))

    print('best validation accuracy achieved during cross-validation: %f' % best_val)

    # Visualize the cross-validation results
    import math
    import pdb

    # pdb.set_trace()

    x_scatter = [math.log10(x[0]) for x in results]
    y_scatter = [math.log10(x[1]) for x in results]

    # plot training accuracy
    marker_size = 100
    colors = [results[x][0] for x in results]
    plt.subplot(2, 1, 1)
    plt.tight_layout(pad=3)
    plt.scatter(x_scatter, y_scatter, marker_size, c=colors, cmap=plt.cm.coolwarm)
    plt.colorbar()
    plt.xlabel('log learning rate')
    plt.ylabel('log regularization strength')
    plt.title('CIFAR-10 training accuracy')

    # plot validation accuracy
    colors = [results[x][1] for x in results]  # default size of markers is 20
    plt.subplot(2, 1, 2)
    plt.scatter(x_scatter, y_scatter, marker_size, c=colors, cmap=plt.cm.coolwarm)
    plt.colorbar()
    plt.xlabel('log learning rate')
    plt.ylabel('log regularization strength')
    plt.title('CIFAR-10 validation accuracy')
    plt.show()
    # Evaluate the best svm on test set
    y_test_pred = best_svm.predict(X_test)
    test_accuracy = np.mean(y_test == y_test_pred)
    print('linear SVM on raw pixels final test set accuracy: %f' % test_accuracy)
    # Visualize the learned weights for each class.
    # Depending on your choice of learning rate and regularization strength, these may
    # or may not be nice to look at.
    w = best_svm.W[:-1, :]  # strip out the bias
    w = w.reshape(32, 32, 3, 10)
    w_min, w_max = np.min(w), np.max(w)
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    for i in range(10):
        plt.subplot(2, 5, i + 1)

        # Rescale the weights to be between 0 and 255
        wimg = 255.0 * (w[:, :, :, i].squeeze() - w_min) / (w_max - w_min)
        plt.imshow(wimg.astype('uint8'))
        plt.axis('off')
        plt.title(classes[i])


if __name__ == '__main__':
    tune_hyperparameters()
