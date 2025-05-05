import os
from pprint import pprint

import pandas as pd

import preset
from f_utility.io_tools import save_json

datasets = [{'fpath': fpath, 'fname': os.path.basename(fpath)} for fpath in [os.path.join(dpath, fname) for dpath in [preset.dpath_eval_train, preset.dpath_eval_test, preset.dpath_dev_train_test] for fname in os.listdir(dpath)] if os.path.isdir(fpath)]

P_devtrain = 'dev-train'
P_devtest = 'dev-test'
P_evaltrain = 'eval-train'
P_evaltest = 'eval-test'


def smart_cast(s: str):
    try:
        i = int(s)
        return i
    except ValueError:
        try:
            f = float(s)
            return f
        except ValueError:
            return s


class PrepInfo():

    def __init__(self, part):
        self.part = part

        self.machine2items = {}
        if self.part == P_devtrain:
            dpath_part = preset.dpath_dev_train_test

            for machine in os.listdir(dpath_part):
                dpath_part_machine = os.path.join(dpath_part, machine)

                dpath_part_machine_tt = os.path.join(dpath_part_machine, 'train')
                dpath_part_machine_csv = os.path.join(dpath_part_machine, 'attributes_00.csv')

                df = pd.read_csv(os.path.join(dpath_part_machine_csv))

                items = []
                for row in df.to_dict(orient="records"):
                    _, tt, fname = row['file_name'].split('/')

                    if tt == 'train':
                        fpath = os.path.join(dpath_part_machine_tt, fname)
                        del row['file_name']

                        label = 0
                        if '_normal_' in fname:
                            label = 1
                        elif '_anomaly_' in fname:
                            label = -1

                        domain = ''
                        if '_source_' in fname:
                            domain = 'source'
                        elif '_target_' in fname:
                            domain = 'target'

                        item = {'label': label} | {'att': [smart_cast(row[key]) for key in list(row.keys()) if 'v' in key]} | {'fpath': fpath} | {'domain': domain}
                        items.append(item)

                self.machine2items[machine] = items

        elif self.part == P_devtest:
            dpath_part = preset.dpath_dev_train_test

            for machine in os.listdir(dpath_part):
                dpath_part_machine = os.path.join(dpath_part, machine)

                dpath_part_machine_tt = os.path.join(dpath_part_machine, 'test')
                dpath_part_machine_csv = os.path.join(dpath_part_machine, 'attributes_00.csv')

                df = pd.read_csv(os.path.join(dpath_part_machine_csv))

                items = []
                for row in df.to_dict(orient="records"):
                    _, tt, fname = row['file_name'].split('/')

                    if tt == 'test':
                        fpath = os.path.join(dpath_part_machine_tt, fname)
                        del row['file_name']

                        label = 0
                        if '_normal_' in fname:
                            label = 1
                        elif '_anomaly_' in fname:
                            label = -1

                        domain = ''
                        if '_source_' in fname:
                            domain = 'source'
                        elif '_target_' in fname:
                            domain = 'target'

                        item = {'label': label} | {'att': [smart_cast(row[key]) for key in list(row.keys()) if 'v' in key]} | {'fpath': fpath} | {'domain': domain}
                        items.append(item)

                self.machine2items[machine] = items
                print()
                pass


part2machine2info = {}
info_dev_train = {}

# dev train
# dev test
# eval train
# eval test


if __name__ == '__main__':
    pass
    pi_devtrain = PrepInfo(P_devtrain)
    pi_devtest = PrepInfo(P_devtest)

    info = {P_devtrain: pi_devtrain.machine2items,
            P_devtest: pi_devtest.machine2items}

    save_json(info, preset.dpath_info_json)
