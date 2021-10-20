import os
import pickle


class AnalysisStore:
    def __init__(self, key, attribute_list=None, directory='analysis_store/'):
        self._attr = attribute_list
        self._key = key
        self._dir = directory

        if attribute_list is None:
            attribute_list = []
        self.data = {k: None for k in attribute_list}
        self._fpath = os.path.join(self._dir, self._key + '.pkl')

    def add(self, attribute, data):
        assert attribute in self.data
        self.data[attribute] = data

    def save(self):
        if not os.path.exists(self._dir):
            os.mkdir(self._dir)

        with open(self._fpath, 'wb') as f:
            pickle.dump(self.data, f)

    def load(self):
        with open(self._fpath, 'rb') as f:
            self.data = pickle.load(f)
