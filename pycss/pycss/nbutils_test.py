import pytest
import numpy as np
from copy import deepcopy
import nbutils


d = 3
arr = np.arange(d**3).reshape(d, d, d)
shape = arr.shape


def test_ind2sub():
    for i in range(arr.size):
       act = ind2sub(i, np.array(shape))
       exp = np.array(np.unravel_index(i, shape))
       assert (act == exp).all()


def test_sub2ind():
   for i in range(arr.size):
       sub = np.array(np.unravel_index(i, shape))
       act = sub2ind(sub, np.array(shape))
       exp = np.ravel_multi_index(sub, shape)
       assert act == exp


def test_list_multi_indexes():
    shape = np.array([2, 3, 4])
    subs = list_multi_indexes(shape)

    assert isinstance(subs, np.ndarray)
    assert subs.shape == (np.prod(shape), len(shape))
    for i in range(len(subs)):
        assert (subs[i] == np.array(np.unravel_index(i, shape))).all()


def test_find_minima():
    arr = np.ones((10, 10, 10))
    center = tuple(np.array(arr.shape) // 2)
    arr[center] = 0.999
    print(arr)

    isMinimum = nbutils.find_local_minima(arr)
    print(isMinimum)

    assert isMinimum.any()
    assert center == tuple([i[0] for i in np.where(isMinimum)])
