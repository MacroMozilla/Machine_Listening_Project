import os
from pathlib import Path

from f_utility.filesystem_tools import list_all_dirs
from f_utility.io_tools import read_json

dpath_root = str(Path(__file__).resolve().parent)


config = read_json(os.path.join(dpath_root, 'config.json'))
dpath_data = config['dpath_data']


dpath_dev_train_test = os.path.join(dpath_data, '7882613')
dpath_eval_train = os.path.join(dpath_data, '7830345')
dpath_eval_test = os.path.join(dpath_data, '7860847')


dpath_info_json = os.path.join(dpath_data, 'info.json')

if __name__ == '__main__':

    pass


    list_all_dirs(dpath_data)
