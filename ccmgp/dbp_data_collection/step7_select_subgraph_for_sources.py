import os
import networkx as nx

from ccmgp.utils import utils

graph_file = utils.RAW_GRAPH_PATH
graph_dir = utils.GRAPH_DIR

G = nx.read_graphml(graph_file)

lang_pairs = []
for i in range(len(utils.langs)) - 1:
    for j in range(i + 1, len(utils.langs)):
        lang_pairs.append([utils.langs[i], utils.langs[j]])

for sources in lang_pairs:
    # First save the partially aligned graphs for each language pair
    selected_nodes = []
    for n in G.nodes():
        lang = n[:2]
        if lang in sources:
            selected_nodes.append(n)
    newG = G.subgraph(selected_nodes)
    nx.write_graphml(newG, os.path.join(data_dir, '_'.join(sorted(sources)) + '_graph.graphml'))

    # Second save unaligned graphs for each language pair by removing SameAs links
    ebunch = []
    for e in newG.edges(data=True):
        if e[2]['type'] == 'sameAs':
            ebunch.append((e[0], e[1]))
    newG2 = nx.Graph(newG)
    newG2.remove_edges_from(ebunch)
    nx.write_graphml(newG2, os.path.join(data_dir, '_'.join(sorted(sources)) + '_graph_unaligned.graphml'))
