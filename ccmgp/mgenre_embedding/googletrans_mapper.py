import os
import numpy as np
import pandas as pd

import ccmgp.utils.utils as utils
import ccmgp.utils.tag_manager as tagM
from base_mapper import Mapper


class GoogleTransMapper(Mapper):
    """ Mapper based on translated tags using Google Translate """

    def __init__(self, tag_manager, translation_dir):
        """ Constructor
        :param tag_manager: an object of type TagManager
        :param translation_dir: the folder with csv files containing the
        translations; for the paper experiment are provided for download
        """
        self.trans = self._read_trans(translation_dir)
        self.name = self.get_name()
        super(GoogleTransMapper, self).__init__(tag_manager)

    def _read_trans(self, translation_dir):
        """ Read provided translations in a dict per language """
        trans = {}
        for f in os.listdir(translation_dir):
            src_lang = f.split('.')[0]
            trans[src_lang] = {}
            df = pd.read_csv(os.path.join(translation_dir, f))
            for lang in utils.langs:
                df[lang] = df[lang].apply(tagM.TagManager.normalize_tag, prefixed=False, ja=lang=='ja')
                df[lang] = df[lang].apply(lambda t: lang + ':' + t)
            src_lang = df.columns[0]
            df = df.set_index(src_lang)
            df = df.loc[~df.index.duplicated(keep='first')]
            trans[src_lang] = df
        return trans

    def _compute_mapping_tbl(self):
        """ Compute translation table """
        tm = self.tag_manager
        norm_src_tags = [t[:2] + ':' + tagM.TagManager.normalize_tag(t, ja=t[:2]=='ja') for t in tm.source_tags]
        norm_tgt_tags = [t[:2] + ':' + tagM.TagManager.normalize_tag(t, ja=t[:2]=='ja') for t in tm.target_tags]
        tbl = np.zeros((len(tm.source_tags), len(tm.target_tags)))
        for i in range(len(norm_src_tags)):
            st = norm_src_tags[i]
            lang = st[:2]
            # This may happen because the source tags are taken from the graph
            #  while the genres from the evaluation corpus are much less
            #  and only those were translated
            if st not in self.trans[lang].index:
                continue
            trans_tgt_tags = self.trans[lang].loc[st].tolist()
            for tt in trans_tgt_tags:
                if tt in norm_tgt_tags:
                    j = norm_tgt_tags.index(tt)
                    tbl[i, j] = 1
        return tbl
