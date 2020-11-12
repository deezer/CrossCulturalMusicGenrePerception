import numpy as np
import networkx as nx

from ccmgp.mgenre_embedding.base_mapper import Mapper


class GraphDirectMapper(Mapper):
    """ Mapper based on DBpedia ontology translation (sameAs link) """

    def __init__(self, tag_manager, G):
        """ Constructor
        :param tag_manager: an object of type TagManager
        :param G: the music genre ontology
        """
        selected_langs = list(tag_manager.sources)
        selected_langs.append(tag_manager.target)
        selected_nodes = []
        for n in G.nodes:
            lang = n[:2]
            if lang in selected_langs:
                selected_nodes.append(n)
        self.G = G.subgraph(selected_nodes)
        self.name = self.get_name()
        super(GraphDirectMapper, self).__init__(tag_manager)

    def _compute_mapping_tbl(self):
        """ Compute mapping table from source to target tags """
        tm = self.tag_manager
        tbl = np.zeros((len(tm.source_tags), len(tm.target_tags)))
        for i in range(len(tm.source_tags)):
            edges = self.G.edges(tm.source_tags[i], data=True)
            for e in edges:
                if e[2]['type'] == 'sameAs':
                    lang = e[1][:2]
                    if lang == tm.target:
                        j = tm.target_tags.index(e[1])
                        # print(tm.source_tags[i], tm.target_tags[j])
                        tbl[i, j] = 1
        return tbl


class GraphDistanceMapper(Mapper):
    """ Mapper based on DBpedia ontology distance table """

    def __init__(self, tag_manager, G, cutoff=2):
        """ Constructor
        :param tag_manager: an object of type TagManager
        :param G: the music genre ontology
        :param cutoff: maximum shortest path considered
        """
        self.G = G
        self.name = self.get_name()
        self.cutoff = cutoff
        super(GraphDistanceMapper, self).__init__(tag_manager)

    def _compute_mapping_tbl(self):
        """ Compute mapping table from source to target tags """
        tm = self.tag_manager
        tbl = np.zeros((len(tm.source_tags), len(tm.target_tags)))
        # Cutoff is set 2 because in this way we can retrieve the direct
        # translation and the neighbours of that translation
        spaths = dict(nx.all_pairs_shortest_path_length(self.G, cutoff=self.cutoff))
        for i in range(len(tm.source_tags)):
            sg = tm.source_tags[i]
            for j in range(len(tm.target_tags)):
                tg = tm.target_tags[j]
                if sg in spaths and tg in spaths[sg]:
                    d = -spaths[sg][tg]
                else:
                    d = -len(self.G)
                tbl[i, j] = d
        return tbl

