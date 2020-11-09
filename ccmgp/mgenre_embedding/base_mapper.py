import numpy as np


class Mapper():
    def __init__(self, tag_manager):
        self.tag_manager = tag_manager
        self.map_tbl = self._compute_mapping_tbl()

    def _compute_mapping_tbl(self):
        raise NotImplementedError("")

    def predict_scores(self, eval_data):
        norm = np.count_nonzero(eval_data, axis=1)
        return eval_data.dot(self.map_tbl) / norm.reshape(norm.shape[0], 1)

    def save_mapping_tbl(self, path):
        self.map_tbl.to_csv(path, index=True, index_label='source')

    def get_name(self):
        return self.__class__.__name__
