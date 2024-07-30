import random
import numpy as np
import tensorflow as tf


def get_index(lst=None, item=None):
    # output:  item 存在於 lst 中所有位置
    return [index for (index, value) in enumerate(lst) if value == item]

def get_index_per_class(label, classes, lbl_n):
    C_index_lst = [[] for i in range(classes)]
    for i in range(classes):
        y_tr_ind = get_index(lst=label.tolist(), item=i)

        while len(y_tr_ind) > lbl_n:
            idx_lbl = random.sample(y_tr_ind, lbl_n)
            C_index_lst[i].append(idx_lbl)
            for ele in idx_lbl:
                y_tr_ind.remove(ele)
        # print('label:', str(i), '>>>', len(C_index_lst[i]))
        try:
            min_len
        except NameError:
            min_len = len(C_index_lst[i])

        if len(C_index_lst[i]) < min_len:
            min_len = len(C_index_lst[i])
    # print(min_len)
    for i in range(classes):
        C_index_lst[i] = C_index_lst[i][0: min_len]
    return np.array(C_index_lst)

def get_sup_batch(X, y, C_index_lst, labelN, lbl_n):
    sup_img = []
    sup_lbl = []
    for c in range(labelN):
        for n in range(lbl_n):
            sup_img.append(X[C_index_lst[c, 0, n]])
            sup_lbl.append(y[C_index_lst[c, 0, n]])
    return np.array(sup_img), np.array(sup_lbl)

def get_unsup_batch(X, y, C_index_lst, labelN, lbl_n):
    unsup_img = []
    unsup_lbl = []
    for b in range(1, C_index_lst.shape[1]):
        img_ = []
        lbl_ = []
        for c in range(labelN):
            for n in range(lbl_n):
                img_.append(X[C_index_lst[c, b, n]])
                lbl_.append(y[C_index_lst[c, b, n]])
        unsup_img.append(img_)
        unsup_lbl.append(lbl_)
    return unsup_img, unsup_lbl

def load_mnist(labelN, lbl_n):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255.
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255.
    x_train = tf.image.resize(x_train, [32, 32])
    x_test = tf.image.resize(x_test, [32, 32])

    # get train index
    C_index_lst = get_index_per_class(y_train, labelN, lbl_n)

    # split sup and unsup
    sup_img, sup_lbl = get_sup_batch(x_train, y_train, C_index_lst, labelN, lbl_n)  # (10x10, 28, 28, 1) (100,)
    unsup_img, unsup_lbl = get_unsup_batch(x_train, y_train, C_index_lst, labelN, lbl_n)  # (541, 10x10, 28, 28, 1) (541, 10x10)

    # get test index
    C_index_lst_ = get_index_per_class(y_test, labelN, lbl_n)

    # split val and test
    val_img, val_lbl = get_sup_batch(x_test, y_test, C_index_lst_, labelN,  lbl_n)  # (10x10, 28, 28, 1) (100,)
    test_img, test_lbl = get_unsup_batch(x_test, y_test, C_index_lst_, labelN, lbl_n)  # (88, 10x10, 28, 28, 1) (88, 10x10)

    return sup_img, sup_lbl, unsup_img, unsup_lbl, val_img, val_lbl, test_img, test_lbl