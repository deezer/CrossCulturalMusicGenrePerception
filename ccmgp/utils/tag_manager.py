import utils
from sklearn.preprocessing import MultiLabelBinarizer

import MeCab
wakati = MeCab.Tagger("-Owakati")


class TagManager:
    """ Tag Manager class used for normalizing tags and preparing MultilabelBinarizer objects needed in evaluation"""
    _MANAGERS_ = {}

    def __init__(self, sources, target):
        self._sources = sources
        self._target = target
        self.source_tags = [el for source in sources for el in utils.get_tags_for_source(source)]
        self.target_tags = [el for el in utils.get_tags_for_source(target)]
        self._mlb_sources = None
        self._mlb_target = None

    @property
    def sources(self):
        return self._sources

    @property
    def target(self):
        return self._target

    @property
    def mlb_sources(self):
        """ Create a MultiLabelBinarizer from source tags"""
        if self._mlb_sources is None:
            self._mlb_sources = MultiLabelBinarizer(classes=self.source_tags, sparse_output=True)
            self._mlb_sources.fit([[]])
        return self._mlb_sources

    @property
    def mlb_target(self):
        """ Create a MultiLabelBinarizer from target tags"""
        if self._mlb_target is None:
            self._mlb_target = MultiLabelBinarizer(classes=self.target_tags, sparse_output=True)
            self._mlb_target.fit([[]])
        return self._mlb_target

    def transform_for_target(self, df, as_array=False):
        if as_array:
            return self.mlb_target.transform(df).toarray().astype("float32")
        else:
            return self.mlb_target.transform(df)

    def transform_for_sources(self, df, as_array=False):
        if as_array:
            return self.mlb_sources.transform(df).toarray().astype("float32")
        else:
            return self.mlb_sources.transform(df)

    @staticmethod
    def normalize_tag(tag, prefixed=True, ja=False):
        """ Normalize a tag
        :param tag: input tag
        :param prefixed: if the tag is prefixed with the language code
        :param ja: if tag is in Japanese language
        :return: the normalized tag
        """
        if prefixed:
            tag = tag[3:]
        return TagManager._norm_basic(tag, ja=ja)

    @staticmethod
    def normalize_tag_basic(t, prefixed=True):
        if prefixed:
            t = t[3:]
        return t.lower().replace('_', '-').replace(',', '')

    @staticmethod
    def _norm_basic(s, ja=False, asList=False, sort=False):
        """Perform a basic normalization
        -lower case
        -replace special characters by space
        -sort the obtained words
        :param s: the input string / tag
        :param ja: if tag is in Japanese language
        :param asList: if the tag tokens are returned as list or concatenated
        :param sort: if the obtained tokens are sorted before concatenation
        :return: the normalized tag
        """
        split_chars_gr1 = ['_', '-', '/', ',', '・']
        split_chars_gr2 = ['(', ')', "'", "’", ':', '.', '!', '‘', '$']
        s = list(s.lower())
        new_s = []
        for c in s:
            if c in split_chars_gr1:
                new_s.append(' ')
            elif c in split_chars_gr2:
                continue
            else:
                new_s.append(c)
        new_s = ''.join(new_s)
        if ja:
            new_s = wakati.parse(new_s).replace('\n', '').rstrip()
        if sorted or asList:
            words = new_s.split()
            if sort:
                words = sorted(words)
                return ' '.join(words)
            if asList:
                return words
        return new_s

    @classmethod
    def get(cls, sources, target):
        """ Return a tag manager instance for the specific sources and target"""
        sources_key = " ".join(sorted(sources))
        if sources_key not in cls._MANAGERS_ or target not in cls._MANAGERS_[sources_key]:
            m = TagManager(sources, target)
            cls._MANAGERS_.setdefault(sources_key, {})
            cls._MANAGERS_[sources_key][target] = m
        return cls._MANAGERS_[sources_key][target]
