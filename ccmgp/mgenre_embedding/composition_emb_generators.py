import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from transformers import XLMModel, XLMTokenizer, BertTokenizer, BertModel

import ccmgp.utils.utils as utils


class FTEmbeddingGenerator:
    """Tag embedding generator from pre-trained fastText embeddings"""

    def __init__(self, path):
        """Constructor
        :param path: fastText embeddings file in text format
        """
        self.embeddings, self.emb_dim = utils.read_embeddings(path)
        self.estimate_word_freqs()

    def estimate_word_freqs(self, mandelbrot=False):
        """ Estimate word frequencies
            The words are given in descending order in the model vocabulary
            z = Zipf rank in a list of words ordered by decreasing frequency
            f(z, N) = frequency of a word with Zipf rank z in a list of N words
            f(z, N) = approx. 1/z
            :param mandelbrot: if Mandelbrot generalization of Zipf should be used. Then f(z, N) = 1/(z + 2.7) (Word Frequency Distributions By R. Harald Baayen)
        """
        word_ranks = {k: v + 1 for v, k in enumerate(self.embeddings.keys())}
        self.word_freqs = {}
        for w in word_ranks:
            if mandelbrot:
                self.word_freqs[w] = 1 / (word_ranks[w] + 2.7)
            else:
                self.word_freqs[w] = 1 / word_ranks[w]

    def get_avg_embs(self, tags, save_file_path=None):
        """Compute embeddings of multi-word tags as ordinary average of its token/word embeddings
        :param tags: a list of tags
        :param save_file_path: if specified the embeddings will be saved here
        :return: computed embeddings for tags
        """
        embs = {}
        for t in tags:
            embs[t] = np.zeros(self.emb_dim)
            words = t.split()
            for w in words:
                if w in self.embeddings:
                    embs[t] += self.embeddings[w]
            embs[t] /= len(words)
        df_embs = pd.DataFrame(embs).T
        if save_file_path is not None:
            df_embs.to_csv(save_file_path, index_label='words', header=False)
        return df_embs

    def get_wei_avg_embs(self, tags, a=1e-3, save_file_path=None):
        """Compute embeddings of multi-word tags as weighted average of its token/word embeddings (smooth inverse frequency averaging)
        :param tags: a list of tags
        :param a: a model hyper-parameter (see Arora et al. in the paper)
        :param save_file_path: if specified the embeddings will be saved here
        :return: computed embeddings for tags
        """
        def _get_weighted_average(tags):
            embs = {}
            for t in tags:
                embs[t] = np.zeros(self.emb_dim)
                words = t.split()
                for w in words:
                    if w in self.embeddings:
                        embs[t] += a / (a + self.word_freqs[w]) * self.embeddings[w]
                embs[t] /= len(words)
            return embs

        def _compute_pc(df_embs, npc=1):
            """Compute the principal component (see Arora at el. in the paper)
            :param df_embs: the input embeddings
            :param npc: number of principal component, default 1
            :return: the principal component
            """
            svd = TruncatedSVD(n_components=npc, n_iter=7, random_state=0)
            svd.fit(df_embs)
            return svd.components_

        def _remove_pc(df_embs, npc=1):
            """Remove the pc (see Arora at el. in the paper)
            :param df_embs: the input embeddings
            :param npc: number of principal component, default 1
            :return: the normalized embeddings
            """
            pc = _compute_pc(df_embs, npc)
            if npc == 1:
                df_embs_out = df_embs - df_embs.dot(pc.transpose()) * pc
            else:
                df_embs_out = df_embs - df_embs.dot(pc.transpose()).dot(pc)
            return df_embs_out

        embs = _get_weighted_average(tags)
        df_embs = pd.DataFrame(embs).T
        new_embs = _remove_pc(df_embs.to_numpy())
        df_embs = pd.DataFrame(new_embs, columns=df_embs.columns, index=df_embs.index)
        if save_file_path is not None:
            df_embs.to_csv(save_file_path, index_label='words', header=False)
        return df_embs


class TransformerEmbeddingGenerator:
    """Tag embedding generator from multilingual transformer models' lookup table (XLM and mBERT)"""

    def __init__(self, model_type):
        """Constructor
        :param model_type: which model is used, xlm or mbert
        """
        if model_type == 'xlm':
            self.tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-100-1280')
            model = XLMModel.from_pretrained('xlm-mlm-100-1280')
            self.embeddings = model.embeddings.weight
        elif model_type == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
            model = BertModel.from_pretrained('bert-base-multilingual-uncased')
            self.embeddings = model.embeddings.word_embeddings.weight
        self.emb_dim = self.embeddings.shape[1]

    def get_avg_embs(self, tags, save_file_path=None):
        """Compute embeddings of multi-word tags as ordinary average of its token/word embeddings
        :param tags: a list of tags
        :param save_file_path: if specified the embeddings will be saved here
        :return: computed embeddings for tags
        """
        embs = {}
        for t in tags:
            embs[t] = np.zeros(self.emb_dim)
            token_ids = self.tokenizer.encode(t, add_special_tokens=False)
            for tid in token_ids:
                embs[t] += self.embeddings[tid].detach().numpy()
            embs[t] /= len(token_ids)
        df_embs = pd.DataFrame(embs).T
        if save_file_path is not None:
            df_embs.to_csv(save_file_path, index_label='words', header=False)
        return df_embs

    def get_wei_avg_embs(self, tags, a=1e-3, save_file_path=None):
        """Compute embeddings of multi-word tags as weighted average of its token/word embeddings (smooth inverse frequency averaging)
        :param tags: a list of tags
        :param a: a model hyper-parameter (see Arora et al. in the paper)
        :param save_file_path: if specified the embeddings will be saved here
        :return: computed embeddings for tags
        """
        def _get_weighted_average(tags):
            embs = {}
            for t in tags:
                embs[t] = np.zeros(self.emb_dim)
                token_ids = self.tokenizer.encode(t, add_special_tokens=False)
                for tid in token_ids:
                    embs[t] += a / (a + 1 / (tid + 1)) * self.embeddings[tid].detach().numpy()
                embs[t] /= len(token_ids)
            return embs

        def _compute_pc(df_embs, npc=1):
            """Compute the principal component (see Arora at el. in the paper)
            :param df_embs: the input embeddings
            :param npc: number of principal component, default 1
            :return: the principal component
            """
            svd = TruncatedSVD(n_components=npc, n_iter=7, random_state=0)
            svd.fit(df_embs)
            return svd.components_

        def _remove_pc(df_embs, npc=1):
            """Remove the pc (see Arora at el. in the paper)
            :param df_embs: the input embeddings
            :param npc: number of principal component, default 1
            :return: the normalized embeddings
            """
            pc = _compute_pc(df_embs, npc)
            if npc == 1:
                df_embs_out = df_embs - df_embs.dot(pc.transpose()) * pc
            else:
                df_embs_out = df_embs - df_embs.dot(pc.transpose()).dot(pc)
            return df_embs_out

        embs = _get_weighted_average(tags)
        df_embs = pd.DataFrame(embs).T
        new_embs = _remove_pc(df_embs.to_numpy())
        df_embs = pd.DataFrame(new_embs, columns=df_embs.columns, index=df_embs.index)
        if save_file_path is not None:
            df_embs.to_csv(save_file_path, index_label='words', header=False)
        return df_embs
