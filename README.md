# Modeling the Music Genre Perception across Language-Bound Cultures

This repository provides Python code to reproduce the experiments from the article [**Modeling the Music Genre Perception across Language-Bound Cultures**](https://www.aclweb.org/anthology/2020.emnlp-main.293/) presented at the [EMNLP 2020](https://2020.emnlp.org/) conference.

The projects consists of three parts:
- `ccmgp/dbp_data_collection`: collect test corpus for experiments and multilingual music genre ontology from DBpedia (see [DBpedia data collection](#dbpedia-data-collection) for more details).
- `ccmgp/mgenre_embedding`: learn multilingual music genre representations with different approaches (see [Music genre embedding](#music-genre-embedding) for more details).
- `ccmgp/experiments`: model and evaluate cross-lingual music genre annotation (see [Experiments](#experiments) for more details).

We currently support six languages from four language families:
- :gb: English (en)
- :netherlands: Dutch (nl)
- :fr: French (fr)
- :es: Spanish (es)
- :czech_republic: Czech
- :jp: Japanese

## Installation

```bash
git clone https://github.com/deezer/CrossCulturalMusicGenrePerception
cd CrossCulturalMusicGenrePerception
python setup.py install
```

Requirements: numpy, pandas, sklearn, networkx, joblib, torch, SPARQLWrapper, mecab-python3, transformers.

## Reproduce published results
We further explain how to reproduce the results reported in the article (Table 1 to Table 6).

### Download data
Data collected from DBpedia, namely the test corpus and the language-specific music genre ontologies, could change over time. Consequently, we provide for download the version used in the experiments reported in the paper. We also include the music genre representations pre-computed with various approaches, as described in our work. More details about how to collect data from DBpedia and learn music genre embeddings from scratch can be found in [DBpedia data collection](#dbpedia-data-collection) and [Music genre embedding](#music-genre-embedding) respectively.

The data is available for download [here](). After download, the `data` folder must be placed in the root folder containing the cloned code. Otherwise, the constant `DATA_DIR` defined in `ccmgp/utils/utils.py` should be changed accordingly.

The `data` folder contains the following data:
- `[ja|cs|nl|fr|es|en]_entities.txt`: music artists, works and bands from DBpedia in the language identified by the code.
- `musical_items_ids.csv`: mapping of DBpedia-based music items on unique identifiers.
- `filtered_musical_items.csv`: the multilingual test corpus containing DBpedia-based music items with music genre annotations in at least two languagues. This corpus has been filtered by removing music genres which did not appear at least 15 times (to ensure that each music genre appears 5 times in each of the 3 data splits in cross-validation).
- `filtered_dbp_graph.graphml`: the multilingual DBpedia-based music genre ontology in a cleaned version . Tags that were not recognized as proper DBpedia resources and the connected components that did not contain at least a corpus music genre were removed. This ontology contain music genre relations of type *sameAs*, being thus partially aligned between languagues.
- `folds`: the parallel corpus split in 3 folds in a stratified way for each language as target.
- `graphs`: for each pair of languages, language-specific music genre ontologies unaligned (without the *sameAs* relation) and aligned (with the *sameAs* relation), extracted from the complete ontology `filtered_dbp_graph.graphml`. These ontologies are used in retrofitting and by the DBpedia-based cross-lingual mappers (one based on direct translation and one based on graph geodesic distances).
- `composed_embs`: pre-computed music genre representations with composition functions from static word embeddings.
- `transf_embs`: pre-computed music genre representations using contextualized language models (mBERT and XLM).
- `laser_embs`: pre-computed music genre representations using LASER, a universal language agnostic sentence embedding model.
- `retro_embs`: pre-computed music genre representations by retrofitting music genre distributed representations to music genre ontologies.
- `google_trans`: csv files for each language, containing literal translations of music genres from one language to all the other five languages, obtained with Google Translate.

### Experiments

Compute statistics about the test corpus and multilingual DBpedia-based music genre ontology (*Table 1* and *Table 2*).
```bash
cd ccmgp/experiments/
python compute_corpus_graph_statistics.py
```

Reproduce the results presented in *Table 3*:
```bash
cd ccmgp/experiments/
python compute_table3_results.py
```

Reproduce the results presented in *Table 4*:
```bash
cd ccmgp/experiments/
python compute_table3_results.py
```

Reproduce the results presented in *Table 5*:
```bash
cd ccmgp/experiments/
python compute_table3_results.py
```

Reproduce the results presented in *Table 6*:
```bash
cd ccmgp/experiments/
python compute_table3_results.py
```

Expected running time (on a MacBook Pro 2.3GHz dual-core Intel Core i5):

Experiment | *Table 3* | *Table 4* | *Table 5* | *Table 6* |
| :--------: |:--------:|--------:|--------:|--------:|
Time    | 1h10m | 10m | 1h20m | 43m |

## Run pipeline from scratch

### DBpedia data collection

### Music genre embedding

## Cite

Please cite our paper if you use this code in your own work:

```BibTeX
@inproceedings{epure2020multilingual,
  title={Modeling the Music Genre Perception across Language-Bound Cultures},
  author={Epure, Elena V. and Salha, Guillaume and Moussallam, Manuel and Hennequin, Romain},
  booktitle={The 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year={2020}
}
```
