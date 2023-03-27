# -*- coding:utf-8 -*-
# @FileName :numpy_test.py
# @Time :2023/3/27 10:05
# @Author :Xiaofeng

def test_hstack():
    """np.hstack将参数元组的元素数组按水平方向进行叠加"""
    import numpy as np

    arr1 = np.array([[1, 3], [2, 4]])
    arr2 = np.array([[1, 4], [2, 6]])
    res = np.hstack((arr1, arr2))

    print(res)


if __name__ == '__main__':
    test_hstack()