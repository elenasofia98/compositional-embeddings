import numpy as np

from base_model.model import BadExampleException


class ExampleToNumpy:
    def __init__(self, data=None, target=None):
        if (data is not None and target is None) or (data is None and target is not None):
            raise ValueError('Data and target must have the same size')

        if data is None and target is None:
            self.data = []
            self.target = []
        else:
            if len(data) != len(target):
                print(f'len data: {len(data)}, len target: {len(target)}')
                raise ValueError('Data and target must have the same size')

            if data is not None and target is not None:
                self.data = data
                self.target = target
                return

    def add_example(self, example):
        if 'target' not in example or 'data' not in example:
            raise BadExampleException()
        if len(example['data']) != 2:
            raise BadExampleException()

        target, data = example['target'], example['data']
        self.data.append_fixed_len(np.array(data))
        self.target.append_fixed_len(np.array(target))

    def save_numpy_examples(self, path):
        np.savez(path, data=self.data, target=self.target)


class POSAwareExampleToNumpy(ExampleToNumpy):
    def __init__(self, data=None, target=None, target_pos=None, w1_pos=None, w2_pos=None):
        super().__init__(data, target)
        if target_pos is None or w1_pos is None or w2_pos is None:
            self.target_pos = []
            self.w1_pos = []
            self.w2_pos = []
        else:
            self.target_pos = target_pos
            self.w1_pos = w1_pos
            self.w2_pos = w2_pos

    def add_example(self, example):
        if 'target' not in example or 'data' not in example or 'target_pos' not in example or 'w1_pos' not in example or 'w2_pos' not in example:
            raise BadExampleException()
        if len(example['data']) != 2:
            raise BadExampleException()

        self.data.append_fixed_len(np.array(example['data']))
        self.target.append_fixed_len(np.array(example['target']))
        self.target_pos.append_fixed_len(np.array(example['target_pos']))
        self.w1_pos.append_fixed_len(np.array(example['w1_pos']))
        self.w2_pos.append_fixed_len(np.array(example['w2_pos']))

    def save_numpy_examples(self, path):
        np.savez(path, data=self.data, target_pos=self.target_pos, target=self.target, w1_pos=self.w1_pos, w2_pos=self.w2_pos)
