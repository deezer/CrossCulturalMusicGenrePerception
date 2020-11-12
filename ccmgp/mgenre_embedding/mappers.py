import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

import ccmgp.utils.utils as utils
import ccmgp.utils.tag_manager as tm
from base_mapper import Mapper


class EmbsMapper(Mapper):
    """ Mapper based on tag embeddings """

    def __init__(self, tag_manager, emb_dir, emb_type):
        """ Constructor
        :param tag_manager: an object of type TagManager
        :param emb_dir: the folder with embeddings for all languages
        :param emb_type: type of embedding
        """
        print('Loaded embeddings type:', emb_type)
        self.emb_type = emb_type
        all_embs = self._read_embs(emb_dir)
        self.model = pd.DataFrame(all_embs)
        super(EmbsMapper, self).__init__(tag_manager)

    def _read_embs(self, emb_dir):
        """ Read embeddings from the given folder """
        raise NotImplementedError("")

    def _compute_mapping_tbl(self):
        """ Compute mapping table from source to target tags """
        tm = self.tag_manager
        tags = self.model.columns
        map_tbl = pd.DataFrame(cosine_similarity(self.model.T), index=tags, columns=tags)
        norm_src_tags = self._normalize_tags(tm.source_tags)
        norm_tgt_tags = self._normalize_tags(tm.target_tags)
        tbl = map_tbl.loc[norm_src_tags, norm_tgt_tags]
        return tbl

    def _normalize_tags(self, tags):
        """ Normalize tags """
        norm_tags = []
        for t in tags:
            norm_t = t[:2] + ':' + tm.TagManager.normalize_tag(t, ja=(t[:2] == 'ja'))
            norm_tags.append(norm_t)
        return norm_tags


class CompositionEmbsMapper(EmbsMapper):
    """ Mapper based on compositional tag embeddings """

    def _read_embs(self, emb_dir):
        all_embs = {}
        self.name = ''.join([self.get_name(), '_', self.emb_type])
        for lang in utils.langs:
            lang_embs_path = os.path.join(emb_dir, lang, self.emb_type + '.csv')
            embs, _ = utils.read_embeddings(lang_embs_path, sep=',', binary=True)
            all_embs.update({lang + ':' + k: embs[k] for k in embs})
        return all_embs


class RetrofitEmbsMapper(EmbsMapper):
    """ Mapper based on retrofitted pre-computed tag embeddings """

    def _read_embs(self, emb_file):
        file_name = os.path.basename(emb_file).split('.')[0]
        self.name = ''.join([self.get_name(), '_', file_name])
        all_embs, _ = utils.read_embeddings(emb_file, sep=',', binary=True)
        return all_embs


class LaserEmbsMapper(EmbsMapper):
    """ Mapper based on LASER tag embeddings """

    def _read_embs(self, emb_dir):
        all_embs = {}
        self.name = self.get_name()
        for lang in utils.langs:
            lang_embs_path = os.path.join(emb_dir, lang, self.emb_type + '.csv')
            embs, _ = utils.read_embeddings(lang_embs_path, sep=',', binary=True)
            all_embs.update({lang + ':' + k: embs[k] for k in embs})
        return all_embs


class TransformerEmbsMapper(EmbsMapper):
    """ Mapper based on contextual tag embeddings """

    def _read_embs(self, emb_file):
        file_name = os.path.basename(emb_file).split('.')[0]
        self.name = ''.join([self.get_name(), '_', file_name])
        all_embs, _ = utils.read_embeddings(emb_file, sep=',', binary=True)
        return all_embs
