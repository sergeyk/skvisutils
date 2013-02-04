import os
from os.path import join, exists
from skpyutils.util import makedirs


class Config(object):
    """
    Config class defines paths and miscellaneous dataset and method attributes.

    Initialize with paths to results and temp directories, and optionally,
    to PASCAL data.

    Parameters
    ----------
    res_dir: string
        Path to directory in which to store results.

    temp_dir: string
        Path to directory in which to store large temporary files.
        Should probably be on a temp filespace, and so is probably
        not propagated between machines.
    
    pascal_dir: string, optional
        path to PASCAL VOC data
    """

    VOCyear = '2007'

    pascal_classes = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
        'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

    def __init__(self, res_dir, temp_dir, pascal_dir=None):
        self.mod_dir = os.path.abspath(os.path.dirname(__file__))
        self.res_dir = makedirs(res_dir)
        self.temp_dir = makedirs(temp_dir)
        self.pascal_dir = pascal_dir

    def __repr__(self):
        s = "Config:\n"
        s += "- res_dir:\t{0}\n".format(self.res_dir)
        s += "- temp_dir:\t{0}\n".format(self.temp_dir)
        s += "- pascal_dir:\t{0}\n".format(self.pascal_dir)
        return s

    def get_pascal_path(self, name):
        """
        Return path to data-split file containing image names.

        Parameters
        ----------
        name: string
            in ['train', 'trainval', 'train', 'test']
        
        Returns
        -------
        dirname: string

        Raises
        ------
        ValueError
        """
        if not exists(self.pascal_dir):
            raise ValueError("self.pascal_dir is not set correctly")
        return join(self.pascal_dir, 'ImageSets/Main/{0}.txt'.format(name))

    def get_cached_dataset_filename(self, name):
        dirname = makedirs(join(self.res_dir, 'cached_datasets'))
        filename = "{}_{}.pickle".format(str(self.VOCyear), name)
        return join(dirname, filename)
    
    @property
    def temp_res_dir(self):
        return makedirs(join(self.temp_dir, 'temp_results'))

    def get_dataset_stats_dir(self, dataset):
        return makedirs(join(self.res_dir, 'dataset_stats', dataset.name))

# def get_dp_dets_filename(dp,train=False):
#     return join(get_evals_dp_dir(dp,train), 'cached_dets.npy')

# def get_dp_clses_filename(dp,train=False):
#     return join(get_evals_dp_dir(dp,train), 'cached_clses.npy')
