import ccmgp.utils.utils as utils


class DataHelper:
    """ Data helper class to load and prepare data for evaluation"""
    def __init__(self, tag_manager, dataset_path=None):
        self.tag_manager = tag_manager
        self.dataset_path = dataset_path
        self.dataset_df = None
        if self.dataset_path is not None:
            print("Loading dataset...")
            self.dataset_df = utils.load_tag_csv(self.dataset_path)
            print("Loaded.")

    def get_test_data(self, fold, as_array=True):
        """Return data for evaluation"""
        return self._get_dataset_split(fold, as_array)

    def _get_dataset_split(self, fold, as_array):
        """Get dataset split and formatted"""
        bool_index = self.dataset_df.fold == fold
        df = self.dataset_df[bool_index]
        train_data, target_data = self._format_dataset_rows_and_split(df, self.tag_manager.sources, self.tag_manager.target)
        return self.transform_sources_and_target_data(train_data, target_data, as_array)

    def _format_dataset_rows_and_split(self, df, sources, target):
        """Format and split dataset rows in train and target data for evaluation"""
        train_data = []
        target_data = []
        for t in zip(*[df[s] for s in list(sources) + [target]]):
            stags = t[:len(sources)]
            ttags = t[-1]
            train_data.append([])
            for i, s in enumerate(sources):
                train_data[-1].extend(self._format_tags_for_source(stags[i], s))
            target_data.append(self._format_tags_for_source(ttags, target))
        return train_data, target_data

    def _format_tags_for_source(self, tags, source):
        """Append source code to each tag"""
        return [source + ":" + t for t in tags]

    def transform_sources_and_target_data(self, source_df, target_df, as_array):
        """Transforms the source and target data using a tag manager object"""
        return self.tag_manager.transform_for_sources(source_df, as_array), self.tag_manager.transform_for_target(target_df, as_array)
