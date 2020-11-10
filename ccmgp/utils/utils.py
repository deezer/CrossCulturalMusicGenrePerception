import re
import ast
from collections import Counter

import numpy as np
import pandas as pd
import networkx as nx

import MeCab
wakati = MeCab.Tagger("-Owakati")

# Relative path to the data folder
DATA_DIR = '../../data/'
# Relative path to the filtered music genre graph
GRAPH_PATH = ''.join([DATA_DIR, 'filtered_dbp_graph.graphml'])
# Relative path to the crawled, unprocessed music genre graph
RAW_GRAPH_PATH = ''.join([DATA_DIR, 'dbp_multigraph.graphml'])
# Relative path to the folders where ontologies for each pair of languages are saved as networkx graphs
GRAPH_DIR = ''.join([DATA_DIR, 'graphs/'])
# Relative path to the cleaned corpus
CORPUS_FILE_PATH = ''.join([DATA_DIR, 'filtered_musical_items.csv'])
# Relative path to the unprocessed corpus file
RAW_CORPUS_FILE_PATH = ''.join([DATA_DIR, 'musical_items.csv'])
# Relative path to the folder containing the data folds
FOLDS_DIR = ''.join([DATA_DIR, '/folds/'])
# Relative path to the folder with multilingual embeddings based on fastText
COMP_FT_EMB_DIR = ''.join([DATA_DIR, 'composed_embs/ft/'])
# Relative path to the folder with multilingual embeddings based on the XLM's lookup table
COMP_XLM_EMB_DIR = ''.join([DATA_DIR, 'composed_embs/xlm/'])
# Relative path to the folder with multilingual embeddings based on the mBERT's lookup table
COMP_BERT_EMB_DIR = ''.join([DATA_DIR, 'composed_embs/bert/'])
# Relative path to the folder with multilingual embeddings based on LASER
LASER_EMB_DIR = ''.join([DATA_DIR, 'laser_embs/'])
# Relative path to the folder with multilingual embeddings obtained with contextual language models
TRANSFORM_EMB_DIR = ''.join([DATA_DIR, 'transf_embs/'])
# Relative path to the folder with multilingual embeddings obtained with retrofitting
RETRO_EMB_DIR = ''.join([DATA_DIR, 'retro_embs/'])
# Relative path to the folder containing literal translations of music genres obtained with Google translate
GOOGLETRANS_DIR = ''.join([DATA_DIR, 'google_trans/'])

# Graph will be loaded only once
GRAPH = None
# Tags per language based on the graph
TAG_PER_LANG = None
# the number of folds in cross-validation
NO_FOLDS = 3

# Languages supported
langs = ['en', 'nl', 'fr', 'es', 'cs', 'ja']
# Strategies to compute multi-word expression embeddings from word/token embeddings
emb_composition_types = ['avg', 'wavg']
# Strategies in retrofitting; weighted is when edges are treated differently in the retrofitting algorithm depending on their types (see paper for detail)
retro_emb_types = ['weighted', 'unweighted']
# DBpedia types of music genre relations
rels_types = {'wikiPageRedirects', 'stylisticOrigin', 'musicSubgenre', 'musicFusionGenre', 'derivative', 'sameAs'}
# DBpedia types of music genre relations which signify equivalence
equiv_rels_types = {'wikiPageRedirects', 'sameAs'}

# DBpedia music genre types per language (their names are translated)
rels = {}
rels['en'] = ['http://dbpedia.org/ontology/wikiPageRedirects',
    'http://dbpedia.org/ontology/stylisticOrigin',
    'http://dbpedia.org/ontology/musicSubgenre',
    'http://dbpedia.org/ontology/derivative',
    'http://dbpedia.org/ontology/musicFusionGenre']
rels['fr'] = ['http://dbpedia.org/ontology/wikiPageRedirects',
    'http://fr.dbpedia.org/property/originesStylistiques',
    'http://fr.dbpedia.org/property/sousGenres',
    'http://fr.dbpedia.org/property/genresDérivés',
    'http://fr.dbpedia.org/property/genresAssociés']
rels['es'] = ['http://dbpedia.org/ontology/wikiPageRedirects',
    'http://es.dbpedia.org/property/origenMusical',
    'http://es.dbpedia.org/property/subgéneros',
    'http://es.dbpedia.org/property/derivados',
    'http://es.dbpedia.org/property/fusiones']
rels['nl'] = ['http://dbpedia.org/ontology/wikiPageRedirects']
rels['cs'] = ['http://dbpedia.org/ontology/wikiPageRedirects',
    'http://dbpedia.org/ontology/stylisticOrigin',
    'http://dbpedia.org/ontology/musicSubgenre',
    'http://dbpedia.org/ontology/derivative',
    'http://dbpedia.org/ontology/musicFusionGenre',
    'https://cs.dbpedia.org/property/podstyly',
    'https://cs.dbpedia.org/property/původVeStylech',
    'https://cs.dbpedia.org/property/směsStylů',
    'https://cs.dbpedia.org/property/odvozenéStyly']
rels['ja'] = ['http://dbpedia.org/ontology/wikiPageRedirects',
    'http://ja.dbpedia.org/property/subgenres',
    'http://ja.dbpedia.org/property/derivatives',
    'http://ja.dbpedia.org/property/fusiongenres',
    'http://ja.dbpedia.org/property/stylisticOrigins']

# Mapping DBpedia music genre relations to their English names
rels_mapping = {}
rels_mapping['http://dbpedia.org/ontology/wikiPageRedirects'] = 'wikiPageRedirects'
rels_mapping['http://dbpedia.org/ontology/stylisticOrigin'] = 'stylisticOrigin'
rels_mapping['http://dbpedia.org/ontology/musicSubgenre'] = 'musicSubgenre'
rels_mapping['http://dbpedia.org/ontology/derivative'] = 'derivative'
rels_mapping['http://dbpedia.org/ontology/musicFusionGenre'] = 'musicFusionGenre'
rels_mapping['http://fr.dbpedia.org/property/originesStylistiques'] = 'stylisticOrigin'
rels_mapping['http://fr.dbpedia.org/property/sousGenres'] = 'musicSubgenre'
rels_mapping['http://fr.dbpedia.org/property/genresDérivés'] = 'derivative'
rels_mapping['http://fr.dbpedia.org/property/genresAssociés'] = 'musicFusionGenre'
rels_mapping['http://es.dbpedia.org/property/origenMusical'] = 'stylisticOrigin'
rels_mapping['http://es.dbpedia.org/property/subgéneros'] = 'musicSubgenre'
rels_mapping['http://es.dbpedia.org/property/derivados'] = 'derivative'
rels_mapping['http://es.dbpedia.org/property/fusiones'] = 'musicFusionGenre'
rels_mapping['https://cs.dbpedia.org/property/původVeStylech'] = 'stylisticOrigin'
rels_mapping['https://cs.dbpedia.org/property/podstyly'] = 'musicSubgenre'
rels_mapping['https://cs.dbpedia.org/property/odvozenéStyly'] = 'derivative'
rels_mapping['https://cs.dbpedia.org/property/směsStylů'] = 'musicFusionGenre'
rels_mapping['http://ja.dbpedia.org/property/stylisticOrigins'] = 'stylisticOrigin'
rels_mapping['http://ja.dbpedia.org/property/subgenres'] = 'musicSubgenre'
rels_mapping['http://ja.dbpedia.org/property/derivatives'] = 'derivative'
rels_mapping['http://ja.dbpedia.org/property/fusiongenres'] = 'musicFusionGenre'


def get_ent_name(ent):
    """Extract the name of a DBpedia entity from its URL
    :param ent: the DBpedia URL of the entity
    :return: the entity name
    """
    tokens = re.findall(r"(?:\w{2}:)?(?:https?:\/\/\w{0,2}.?dbpedia.org\/resource\/)(.+(?!_)[\w\!])(?:$|(_?\(.+\)$))", ent)
    if len(tokens) == 0:
        return None
    return tokens[0][0]


def get_lang(ent):
    """Extract the language from a DBpedia entity URL
    :param ent: the DBpedia entity url
    :return: the language code
    """
    if ent.startswith('http://dbpedia.org'):
        return 'en'
    tokens = re.findall(r'(?:https?://)(.{2})(?:\..+)', ent)
    if len(tokens) > 0:
        return tokens[0]
    return None


def get_endpoint_for_lang(lang):
    """ Return the DBpedia endpoint for a specific language
    :param lang: the language of the DBpedia which is queried
    :return: the endpoint
    """
    if lang == 'en':
        endpoint = "http://dbpedia.org/"
    elif lang == 'cs':
        endpoint = "https://cs.dbpedia.org/"
    else:
        endpoint = "http://[LANG].dbpedia.org/".replace('[LANG]', lang)
    return endpoint


def get_genre_keyword(lang):
    """ Return the genre keyword in DBpedia, which varies with the language
    :param lang: the language of the DBpedia which is queried
    :return: the genre keyword
    """
    if lang == 'cs':
        return 'style'
    return 'genre'


def get_alias_filter(lang, langs):
    """ Format the part of query which retrieves aliases only in the languages of interest
    :param lang: the language of the DBpedia which is queried
    :param langs: the list of targetted languages
    :return: the formatted part of query which will be joined to the main query
    """
    other_langs_cond = ''
    for other_lang in langs:
        if other_lang == lang:
            continue
        if other_lang == 'en':
            other_langs_cond += 'strstarts(str(?alias), "http://dbpedia.org/") || '
        else:
            other_langs_cond += ''.join(['strstarts(str(?alias), "http://', other_lang, '.dbpedia.org/") || '])
    other_langs_cond = other_langs_cond[:-4]
    return other_langs_cond


def get_genre_rels_filter(lang):
    """ Helper to format the part of the query which retrieves music genres by crawling genre relations
    :param lang: the language of the DBpedia which is queried
    :return: the formatted part of query which will be joined to the main query
    """
    cond = ''
    for i in range(len(rels[lang])):
        if i == len(rels[lang]) - 1:
            cond += ''.join(['<', rels[lang][i], '>'])
        else:
            cond += ''.join(['<', rels[lang][i], '>', ', '])
    return cond


def get_seeds_filter(seeds):
    """Helper to format the part of the query which retrieves music genres for a seed list of music items provided through their URLs
    :param seeds: seed music items
    :return: the formatted part of query which will be joined to the main query
    """
    list_genres_str = ''
    for g in seeds:
        if not g.startswith('http'):
            continue
        list_genres_str += ''.join(['<', g, '>, '])
    list_genres_str = list_genres_str[:-2]
    return list_genres_str


def corpus_genres_per_lang(df, min_count=1):
    """Get corpus genres per language which appear at least min_count times
    :param df: the corpus
    :param min_count: number of times a genre should appear, default 1
    :return: the genres per language that appear at least min_count times
    """
    selected_tags = {}
    for lang in langs:
        tags = []
        for annotations in df[lang].dropna().tolist():
            tags.extend(ast.literal_eval(str(annotations)))
        counter = Counter(tags)
        selected_tags[lang] = set()
        for x in counter:
            if counter[x] >= min_count:
                selected_tags[lang].add(x)
    return selected_tags


def all_formatted_genres(df, norm_tags=True, as_set=True):
    """Get corpus music genre names
    :param df: the corpus dataframe
    :param norm_tags: specifies if tags are normalized or not
    :param as_set: specifies if the results is a dictionary with genres per language or a set containing all multilingual genres
    :return: the corpus music genre names
    """
    genres = corpus_genres_per_lang(df)
    all_genres = {}
    for lang in genres:
        all_genres[lang] = set()
        for g in genres[lang]:
            if norm_tags:
                g_name = get_ent_name(g)
            else:
                g_name = g
            all_genres[lang].add(g_name)
    if as_set:
        all_genres_set = set()
        for lang in all_genres:
            for g in all_genres[lang]:
                all_genres_set.add(''.join([lang + ':' + g]))
        return all_genres_set
    return all_genres


def get_tags_for_source(source):
    """Get unique music genres in the multilingual graph for a source
    :param source: the language
    :param graph_path: the graph file path
    :return: tags per source / language
    """
    global GRAPH
    global TAG_PER_LANG
    if TAG_PER_LANG is None or source not in TAG_PER_LANG:
        if GRAPH is None:
            GRAPH = nx.read_graphml(GRAPH_PATH)
        TAG_PER_LANG = {}
        if source not in TAG_PER_LANG:
            for node in GRAPH:
                lang = node[:2]
                if lang not in TAG_PER_LANG:
                    TAG_PER_LANG[lang] = []
                TAG_PER_LANG[lang].append(node)
    return TAG_PER_LANG[source]


def get_graph():
    """Returns the multilingual DBpedia-based music genre graph
    :param graph_path: the graph file path
    """
    global GRAPH
    if GRAPH is None:
        GRAPH = nx.read_graphml(GRAPH_PATH)
    return GRAPH


def load_tag_csv(path, sources=langs, sep='\t'):
    """Load a tag csv in a dataframe
    :param path: the dataset file path
    :param langs: the columns mapped on languages
    :param sep: the separator in the data file
    :return: a dataframe with the data
    """
    df = pd.read_csv(path, sep=sep)

    def load_row(r):
        if isinstance(r, float):
            return []
        else:
            return eval(r)

    def format_values(r):
        formatted_r = []
        for v in r:
            formatted_r.append(get_ent_name(v))
        return formatted_r

    for source in sources:
        df[source] = df[source].apply(load_row)
        df[source] = df[source].apply(format_values)
    return df


def read_embeddings(path, sep=' ', binary=False):
    """Read embeddings given in text format
    :param path: the embedding file path
    :param sep: the separator used in the file, default space
    :return: the embeddings as a dict and their dimension
    """
    if binary:
        return read_binary_embeddings(path)
    embeddings = {}
    emb_dim = None
    with open(path, 'r', encoding='utf-8') as _:
        for line in _:
            values = line.rstrip().rsplit(sep)
            if len(values) == 2:
                emb_dim = int(values[1])
            else:
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings[word] = coefs
    return embeddings, emb_dim


def csv_to_binary(path):
    """Read embedding dataframe from csv and save it in a binary format
    :param path: the path to the csv
    """
    f = open(path, 'r', encoding='utf-8')
    vecs = []
    path = path.replace('.csv', '')
    with open(path + '.vocab', 'w', encoding='utf-8') as _:
        for line in f:
            values = line.rstrip().rsplit(',')
            _.write(values[0])
            _.write("\n")
            vecs.append([float(val) for val in values[1:]])
    np.save(path + '.npy', np.array(vecs, dtype='float32'))


def read_binary_embeddings(path):
    """ Read embeddings from a binary file
    :param path: the file path
    :return: the embeddings as a dict and their dimension
    """
    path = path.replace('.csv', '')
    with open(path + '.vocab', 'r', encoding='utf-8') as _:
        index2word = [line.rstrip().replace('\n', '') for line in _]
    vecs = np.load(path + '.npy')
    model = {}
    for i, w in enumerate(index2word):
        model[w] = vecs[i]
    return model, vecs.shape[1]

