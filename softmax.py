# -*- coding:utf-8 -*-
# @FileName :softmax.py
# @Time :2023/3/28 14:25
# @Author :Xiaofeng
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from cs231n.data_utils import preprocess_data
from cs231n.classifiers.softmax import softmax_loss_naive
from cs231n.classifiers import Softmax
import time


def softmax_loss_naive_test():
    X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev = preprocess_data()
    # Generate a random softmax weight matrix and use it to compute the loss.
    W = np.random.randn(3073, 10) * 0.0001
    loss, grad = softmax_loss_naive(W, X_dev, y_dev, 0.0)

    # As a rough sanity check, our loss should be something close to -log(0.1).
    print('loss: %f' % loss)
    print('sanity check: %f' % (-np.log(0.1)))


def tune_hyperparameters():
    """
    Use the validation set to tune hyperparameters (regularization strength and
    learning rate). You should experiment with different ranges for the learning
    rates and regularization strengths; if you are careful you should be able to
    get a classification accuracy of over 0.35 on the validation set.
    """

    results = {}
    best_val = -1
    best_softmax = None

    ################################################################################
    # TODO:                                                                        #
    # Use the validation set to set the learning rate and regularization strength. #
    # This should be identical to the validation that you did for the SVM; save    #
    # the best trained softmax classifer in best_softmax.                          #
    ################################################################################
    X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev = preprocess_data()
    # Provided as a reference. You may or may not want to change these hyperparameters
    learning_rates = [1e-8, 3e-7]
    regularization_strengths = [2.5e4, 5e4]
    lr_range = np.linspace(learning_rates[0], learning_rates[1], 8)
    reg_range = np.linspace(regularization_strengths[0], regularization_strengths[1], 8)
    # 定义损失函数训练的模型
    softmax = Softmax()
    tic = time.time()

    for lr in tqdm(lr_range):
        for reg in reg_range:
            loss_hist = softmax.train(X_train, y_train, learning_rate=lr, reg=reg, num_iters=500, verbose=False)
            y_train_pred = softmax.predict(X_train)
            y_val_pred = softmax.predict(X_val)
            acc_train = np.mean(y_train == y_train_pred)
            acc_val = np.mean(y_val == y_val_pred)
            # load result into results
            results[(lr, reg)] = (acc_train, acc_val)
            if acc_val > best_val:
                best_val = acc_val
                best_softmax = softmax
            # Print out results.
            print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
                lr, reg, acc_train, acc_val))

    print('best validation accuracy achieved during cross-validation: %f' % best_val)
    # evaluate on test set
    # Evaluate the best softmax on test set
    y_test_pred = best_softmax.predict(X_test)
    test_accuracy = np.mean(y_test == y_test_pred)
    print('softmax on raw pixels final test set accuracy: %f' % (test_accuracy,))

    # Visualize the learned weights for each class
    w = best_softmax.W[:-1, :]  # strip out the bias
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
    plt.show()


if __name__ == '__main__':
    tune_hyperparameters()
