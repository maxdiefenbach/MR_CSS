import numpy as np
from numba import njit


@njit
def ind2sub(ind, shape):
   """shape needs to be an numpy.array

   :param ind: int
   :param shape: 1d numpy.array
   :returns:
   :rtype: 1d numpy.array

   """
   return [ind // np.prod(shape[j+1:]) % shape[j] for j in range(len(shape))]


@njit
def sub2ind(sub, shape):
   """input needs to be numpy.arrays

   :param sub: 1d numpy.array
   :param shape: 1d numpy
   :returns:
   :rtype: int

   """
   ndim = len(shape)
   ind = 0
   shift = 1
   for i in range(ndim-1, -1, -1):
      ind += shift * sub[i]
      shift *= shape[i]
   return ind


@njit
def list_multi_indexes(shape):
   """list all multi-indices ("ind2sub's") for a given array shape

   :param shape: 1d numpy.array
   :returns:
   :rtype: 2d numpy.ndarray
   """
   N = np.prod(shape)
   res = np.zeros((N, len(shape)), dtype=np.int64)
   for i in range(N):
      res[i] = np.array(ind2sub(i, shape))

   return res


@njit
def find_local_minima(array):
    """find all local minima in the n-dimensional array

    :param array: numpy.ndarray
    :returns:
    :rtype: np.ndarray int (boolean)

    """
    shape = np.array(array.shape)
    ndim = len(shape)
    arr = array.ravel()
    L = len(arr)

    isMinimum = np.ones(L, dtype=np.int64)  # true
    for i in range(L):
        sub = ind2sub(i, shape)

        for d in range(ndim):
            # skip singelton dimensions
            if shape[d] == 1:
                continue

            # neighbor indices in current axis without boundary conditions
            i0 = sub[d] - 1
            i1 = sub[d] + 1

            # neighbor sub's with circular boundary conditions
            sub0 = sub.copy()
            sub1 = sub.copy()
            sub0[d] = i0 if i0 >= 0 else int(shape[d]-1)
            sub1[d] = i1 if i1 < int(shape[d]) else 0

            # neighbor indices in flattended array
            n0 = sub2ind(sub0, shape)
            n1 = sub2ind(sub1, shape)

            if not ((arr[i] < arr[n0]) and (arr[i] < arr[n1])  # current axis
                    and isMinimum[i]):  # other axes
                isMinimum[i] = 0  # false

    return isMinimum.reshape(array.shape)
