import ast
from collections import Counter
from operator import itemgetter
import networkx as nx
import pandas as pd
import numpy as np

import ccmgp.utils.utils as utils
from ccmgp.utils.utils import langs

print("***** CORPUS STATISTICS *****")
df = pd.read_csv(utils.CORPUS_FILE_PATH, index_col='id')
size = len(df)
print('Corpus size ', size)
counts = Counter(df.isnull().sum(axis=1).tolist())
for i in sorted(counts.keys()):
    print('Annotation in ' + str(len(langs) - i) + ' languages:', counts[i])

pair_of_langs = {}
for lang in langs:
    print(lang, len(df[[lang]].dropna()), 'entities annotated')
    pair_of_langs[lang] = {}
    for lang2 in langs:
        if lang == lang2:
            pair_of_langs[lang][lang2] = 0
        else:
            pair_of_langs[lang][lang2] = len(df[[lang, lang2]].dropna())

print(pd.DataFrame(pair_of_langs))

print('Unique genres for each language')
selected_tags = utils.corpus_genres_per_lang(df)
for lang in langs:
    print(lang, len(selected_tags[lang]))

print('Average genres per item and standard deviation')
for lang in langs:
    number_per_item = [len(ast.literal_eval(l)) for l in df[lang].dropna().tolist()]
    print(lang, "%.2f"%np.mean(number_per_item), "%.2f"%np.std(number_per_item))

print("\n***** GRAPH STATISTICS *****")
G = utils.get_graph()
print('Total number of genres in graph', len(G.nodes))
print('Total number of edges in graph', len(G.edges))
ccs = sorted(nx.connected_components(G), key=len, reverse=True)
print('Number of connected components', len(ccs))
print('Size of each connected component')
counter = Counter([len(cc) for cc in ccs])
print(sorted(counter.items(), key=itemgetter(1), reverse=False))
for lang in utils.langs:
    tags = utils.get_tags_for_source(lang)
    print('Genre in', lang, ':', len(tags))
