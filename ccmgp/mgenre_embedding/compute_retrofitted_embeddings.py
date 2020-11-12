import os
import pandas as pd
import numpy as np
import networkx as nx
from copy import deepcopy

import ccmgp.utils.utils as utils
import ccmgp.utils.tag_manager as tm


class RetroEmbeddingGenerator:
    """Tag embedding generator by retrofitting pre-computed embeddings to tag
    ontologies (or learning them from scratch through retrofitting)"""

    def __init__(self, input_embs_dir, graph_file, ignore_langs=[]):
        """Contructor
        :param input_embs_dir: the folder with pre-computed embeddings
        :param graph_file: the graph file of type graphml
        :param ignore_langs: used when learning embedding from scratch, the
        embeddings for these languages are considered 0
        """
        # Read pre-computed embeddings
        self.embeddings = {}
        for lang in utils.langs:
            emb_path = os.path.join(input_embs_dir, lang, 'wavg.csv')
            embs, _ = utils.read_embeddings(emb_path, sep=',', binary=False)
            if lang in ignore_langs:
                print(lang)
                self.embeddings.update({lang + ':' + k: np.zeros(embs[k].shape) for k in embs})
            else:
                self.embeddings.update({lang + ':' + k: embs[k] for k in embs})
        if ignore_langs != []:
            self.name = ''.join([os.path.basename(graph_file).split('.')[0], '_unknown_', '_'.join(ignore_langs)])
        else:
            self.name = os.path.basename(graph_file).split('.')[0]
        # Read graph
        self._read_graph(graph_file)

    def _read_graph(self, graph_file):
        """ Read nx graph and normalize nodes
        :param graph_file: the graph file of type graphml
        """
        G = nx.read_graphml(graph_file)
        edges_to_remove = list(G.selfloop_edges())
        G.remove_edges_from(edges_to_remove)
        mapping = {}
        for n in G.nodes:
            mapping[n] = n[:2] + ':' + tm.TagManager.normalize_tag(n, prefixed=True, ja=n[:2]=='ja')
        self.G = nx.relabel_nodes(G, mapping)

    def generate_embs(self, output_embs_dir):
        """ Generate embeddings with retrofitting
        :param output_embs_dir: where the embeddings are saved
        """
        def _beta_f_weighted(i, j, edges):
            """ Function used in retrofitting to weight how much each neighbour
            embedding should count in the iterative update by considering the
            type of the relation between the two nodes,i and j, in the ontology
            """
            for et in utils.equiv_rels_types:
                if j in edges[et][i]:
                    return 1
            count = 0
            for et in utils.rels_types:
                count += len(edges[et][i])
            return 1 / count if count > 0 else 0

        embs = pd.DataFrame(self.embeddings).T
        exprs = list(self.embeddings.keys())
        mapping = dict(zip(exprs, list(range(len(exprs)))))
        known_exprs_ids = self._get_known_exprs_ids(mapping)
        undirect_egdes = self._get_undirected_edges(mapping)
        y = self._retrofit_identity(embs.values, undirect_egdes, known_exprs_ids, beta=_beta_f_weighted)
        retro_embs = pd.DataFrame(y, index=exprs)
        file_name = '_'.join(['wavg', self.name, 'weighted']) + '.csv'
        retro_embs.to_csv(os.path.join(output_embs_dir, file_name), index_label='words', header=None)

    def _get_known_exprs_ids(self, mapping):
        """ Check if a multi-word expression is composed of known vocabulary
        words / tokens; if it is not known, its embedding values are 0"""
        known_ids = {}
        for w in self.embeddings:
            known_ids[mapping[w]] = (np.sum(self.embeddings[w]) == 0)
        return known_ids

    def _get_undirected_edges(self, mapping):
        """ Get all undirected edges from ontology as a dict per relation types
        """
        edges = {}
        for et in utils.rels_types:
            edges[et] = {}
            for g in self.G.nodes:
                edges[et][mapping[g]] = []
        for s, t, meta in self.G.edges(data=True):
            edges[meta['type']][mapping[s]].append(mapping[t])
            edges[meta['type']][mapping[t]].append(mapping[s])
        return edges

    def _retrofit_identity(self, X, edges, known, beta, n_iter=100, tol=1e-2, verbose=True):
        """ Implement the retrofitting method of Faruqui et al.
        :param X: distributional embeddings
        :param edges: edge dict; if multiple types of edges,
            this will be flattened.
        :param known: ontology nodes which have initial embeddings known (different than 0)
        :param beta: see _beta_f_weighted func
        :param n_iter: the maximum number of iterations to run
        :param tol: if the average distance change between two rounds is at or
            below this value, it stops
        :return: Y the retrofitted embeddings
    """
        def alpha(i, known):
            return 1 if i in known else 0
        edges_unflattened = deepcopy(edges)
        if isinstance(next(iter(edges.values())), dict):
            edges = self._flatten_edges(edges, len(X))
        Y = X.copy()
        Y_prev = Y.copy()
        for iteration in range(1, n_iter + 1):
            if verbose:
                print("Iteration", iteration, "of", n_iter)
            for i, vec in enumerate(X):
                neighbors = edges[i]
                n_neighbors = len(neighbors)
                if n_neighbors:
                    a = alpha(i, known)
                    retro = np.array([(beta(i, j, edges_unflattened) + beta(j, i, edges_unflattened)) * Y[j] for j in neighbors])
                    retro = retro.sum(axis=0) + (a * X[i])
                    norm = np.array([beta(i, j, edges_unflattened) + beta(j, i, edges_unflattened) for j in neighbors])
                    norm = norm.sum(axis=0) + a
                    Y[i] = retro / norm
            changes = np.abs(np.mean(np.linalg.norm(
                np.squeeze(Y_prev)[:1000] - np.squeeze(Y)[:1000], ord=2)))
            if changes <= tol:
                if verbose:
                    print("Converged at iteration {}".format(iteration))
                return Y
            else:
                Y_prev = Y.copy()
        if verbose:
            print("Stopping at iteration {:d}; change was {:.4f}".format(iteration, changes))
        return Y

    def _flatten_edges(self, edges, n_nodes):
        """ Flatten a dict of lists of edges of different types.
        :param edges: maps edge type to dict that maps index to neighbors
        :param n_nodes: the number of nodes in the graph.
        :return edges: dict that maps index to all neighbors
        """
        edges_naive = {}
        for i in range(n_nodes):
            edges_naive[i] = []
            for rel_name in edges.keys():
                edges_r = edges[rel_name]
                try:
                    my_edges = edges_r[i]
                except KeyError:
                    continue
                edges_naive[i].extend(my_edges)
        return edges_naive


input_embs_dir = utils.COMP_FT_EMB_DIR
output_embs_dir = utils.RETRO_EMB_DIR
if not os.path.exists(output_embs_dir):
    os.makedirs(output_embs_dir)

lang_pairs = []
for i in range(len(utils.langs) - 1):
    for j in range(i + 1, len(utils.langs)):
        lang_pairs.append(sorted([utils.langs[i], utils.langs[j]]))

for pair in lang_pairs:
    print('Language pair', pair)
    # Retrofit on language-specifi, unaligned graphs
    graph_file = ''.join([utils.GRAPH_DIR, pair[0], '_', pair[1], "_graph_unaligned.graphml"])
    retro_emb_generator = RetroEmbeddingGenerator(input_embs_dir, graph_file, ignore_langs=[])
    retro_emb_generator.generate_embs(output_embs_dir)

    # Retrofit on partially aligned graphs with the goal to learn embeddings
    # for one lang from scratch by knowing the embdddings of the other lang
    graph_file = ''.join([utils.GRAPH_DIR, pair[0], '_', pair[1], "_graph.graphml"])
    retro_emb_generator = RetroEmbeddingGenerator(input_embs_dir, graph_file, ignore_langs=[pair[0]])
    retro_emb_generator.generate_embs(output_embs_dir)
    retro_emb_generator = RetroEmbeddingGenerator(input_embs_dir, graph_file, ignore_langs=[pair[1]])
    retro_emb_generator.generate_embs(output_embs_dir)
