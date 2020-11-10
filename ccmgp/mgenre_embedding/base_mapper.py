import numpy as np


class Mapper():
    """ Base Mapper class"""
    def __init__(self, tag_manager):
        """ Contructor, with an object of type TagManager as param
        """
        self.tag_manager = tag_manager
        self.map_tbl = self._compute_mapping_tbl()

    def _compute_mapping_tbl(self):
        """ Compute mapping table from source tags to target tags
        """
        raise NotImplementedError("")

    def predict_scores(self, eval_data):
        """ Predict scores from eval_data, used in evaluation
        """
        norm = np.count_nonzero(eval_data, axis=1)
        return eval_data.dot(self.map_tbl) / norm.reshape(norm.shape[0], 1)

    def save_mapping_tbl(self, path):
        """Save mapping table as csv file
        """
        self.map_tbl.to_csv(path, index=True, index_label='source')

    def get_name(self):
        """Return the mapper name, default the parent class name Mapper
        """
        return self.__class__.__name__
