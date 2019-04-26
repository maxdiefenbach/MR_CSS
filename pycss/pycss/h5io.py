import os
import h5py as h5
import numpy as np


def load_ImDataParams_mat(filename):
    """load a "*ImDataParams.mat" file save from MATLAB
    and return it as a python dict

    :param filename:
    :returns:
    :rtype:

    """
    print(f'Load {filename}... ', end='')
    with h5.File(filename, 'r') as f:
        attrs_dict = recursively_load_attrs(f)
        data_dict = recursively_load_data(f, attrs_dict)

    nested_dict = nest_dict(data_dict)

    nested_dict['ImDataParams'] = nest_dict(nested_dict['ImDataParams'])
    nested_dict['ImDataParams']['VersionInfo'] = \
        nest_dict(nested_dict['ImDataParams']['VersionInfo'])
    nested_dict['ImDataParams']['VersionInfo']['BMRR'] = \
        nest_dict(nested_dict['ImDataParams']['VersionInfo']['BMRR'])

    nested_dict['Version'] = nest_dict(nested_dict['Version'])
    nested_dict['Version']['BMRR'] = \
        nest_dict(nested_dict['Version']['BMRR'])
    print('Done.')

    return nested_dict


def load_WFIparams_mat(filename):
    """load a "*WFIparams.mat" file save from MATLAB
    and return it as a python dict

    :param filename:
    :returns:
    :rtype:

    """
    with h5.File(filename, 'r') as f:
        attrs_dict = recursively_load_attrs(f)
        data_dict = recursively_load_data(f, attrs_dict)

    nested_dict = nest_dict(data_dict)

    return nested_dict


def recursively_load_attrs(h5file, path='/'):
    """
    recursively load attributes for all groups and datasets in
    hdf5 file as python dict

    :param h5file: h5py.File(<filename>, 'r')
    :param path: "directory path" in h5 File
    :returns:
    :rtype: nested dicts

    """

    attrs_dict = {}
    for k, v in h5file[path].items():

        d = {}
        for ak, av in v.attrs.items():
            d[ak] = av

        if isinstance(v, h5._hl.dataset.Dataset):
            attrs_dict[k] = d

        elif isinstance(v, h5._hl.group.Group):
            d.update(recursively_load_attrs(
                    h5file, os.path.join(path, k)))
            attrs_dict[k] = d

    return attrs_dict


def recursively_load_data(h5file, attrs_dict, path='/'):
    """
    recursively load data for all groups and datasets in
    hdf5 file as python dict corresponding to attrs_dict
    (see function recursively_load_attrs)

    :param h5file: h5py.File(<filename>, 'r')
    :param attrs_dict: output of function recursively_load_attrs
    :returns:
    :rtype: nested dicts

    """

    result = {}
    for k, v in attrs_dict.items():

        if k == '#refs#':
            continue

        if isinstance(v, dict):

            if v.get('MATLAB_class') != b'struct':

                val = h5file[path+k+'/'][...]

                if isinstance(val, np.ndarray) and \
                   (val.dtype == [('real', '<f8'), ('imag', '<f8')] or
                    val.dtype == [('real', '<f4'), ('imag', '<f4')]):
                    val = np.transpose(val.view(np.complex))

                if v.get('MATLAB_class') == b'char':
                    val = ''.join([chr(c) for c in h5file[path+k+'/']])

                result[path+k+'/'] = val

            else:
                result.update(recursively_load_data(h5file, v, path+k+'/'))

    return result


def nest_dict(flat_dict):
    seperator = '/'
    nested_dict = {}
    for k, v in flat_dict.items():

        path_list = list(filter(None, k.split(seperator))) # removes '' elements
        split_key = path_list.pop(0)
        left_key = seperator.join(path_list)

        if left_key == '':
            nested_dict[split_key] = v
            continue

        if not nested_dict.get(split_key): # init new dict
            nested_dict[split_key] = {}

        if left_key != '':
            nested_dict[split_key].update({left_key: v})

    return nested_dict
