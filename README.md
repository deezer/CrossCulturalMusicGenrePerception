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
python compute_table3_results.py
```

Reproduce the results presented in *Table 4*:
```bash
python compute_table3_results.py
```

Reproduce the results presented in *Table 5*, found in *Appendix*:
```bash
python compute_table3_results.py
```

Reproduce the results presented in *Table 6*, found in *Appendix*:
```bash
python compute_table3_results.py
```

Expected running time (on a MacBook Pro 2.3GHz dual-core Intel Core i5):

Experiment | *Table 3* | *Table 4* | *Table 5* | *Table 6* |
| :--------: |:--------:|--------:|--------:|--------:|
Time    | 1h10m | 10m | 1h20m | 43m |


As previously mention, DBpedia changes over time. New music artists, works or bands could be added or some of the past ones could be removed. The annotations of music items with music genres could be modified too. Hence, these changes have an impact on the test corpus.
Additionally, the music genre ontology could also evolve because music genres or music genre relations are added to or removed from DBpedia.

For this reason, if experiments are run with new data collected at another moment from DBpedia, the macro-AUC scores may not be identical to the ones reported in the paper. However, we should still reach the same conclusions as those presented in the paper:
- We can model the cross-lingual music genre annotation with high accuracy, especially when combining the two types of language-specific semantic representations, ontologies and distributed embeddings.
- Using literal translation to produce cross-lingual annotations is limited as it does not consider the culturally divergent perception of concepts.
- For short multi-word expressions, when comparing the representations derived from multilingual pre-trained models, the smooth inverse frequency averaging (Arora et al., 2017) of aligned word embeddings outperforms the other state of the art approaches.
- When aligned multilingual concept ontologies are available and concept embeddings in one language are known, embedding learning from scratch with retrofitting for the other language leads to very relevant representations.

## DBpedia data collection
We further explain how to collect data from DBpedia. Each step uses the output of the previous step as input. Therefore, it is important that the previous step finishes correctly. A problem that could appear is that DBpedia in a certain language could be temporarily down. In this case, there are two options:
- wait until DBpedia is again up and could be queried correctly.
- remove the concerned language from `langs` in `utils.py`.

#### Step 1: collect DBpedia music artists, bands and music works
```bash
cd ccmgp/dbp_data_collection
python step1_collect_dbp_music_items.py
```

Input: nothing

Output: `[fr|es|en]_entities.txt` and `musical_items_ids.csv`

#### Step 2: collect DBpedia-based music genres annotations for music items
```bash
python step2_collect_dbp_genres_for_music_items.py
```

Input: `[fr|es|en]_entities.txt` and `musical_items_ids.csv`

Output: `musical_items.csv`

#### Step 3: filter the corpus by removing music genres that do not appear at least 15 times
```bash
python step3_filter_corpus.py
```

Input: `musical_items.csv`

Output: `filtered_musical_items.csv`

#### Step 4: split corpus in 3 folds for each language
```bash
python step4_prepare_folds_eval.py
```

Input: `filtered_musical_items.csv`

Output: the files of type `[fr|es|en]_4-fold.tsv`) in the `folds` folder

#### Step 5 collect the multilingual DBpedia-based music genre ontology
```bash
python step5_collect_dbp_mgenre_ontology.py
```

Input: `filtered_musical_items.csv`

Output: `dbp_multigraph.graphml`

#### Step 6: clean the raw multilingual DBpedia-based music genre ontology
```bash
python step6_clean_dbp_ontology.py
```

Input: `dbp_multigraph.graphml`

Output: `filtered_dbp_graph.graphml`

#### Step 7: extract aligned and unaligned language-specific music genre ontologies
```bash
python step7_select_subgraph_for_sources.py
```
Input: `filtered_dbp_graph.graphml`

Output: the `graphs` folder; for each language pair, the aligned ontology is saved in the `lang1_lang2_graph.graphml` file and the unaligned ontologies are saved in the `lang1_lang2_graph_unaligned.graphml` file.

## Music genre embedding
This part relies on the successful data collection from DBpedia. Further, we show how to generate multilingual music genre embeddings with various strategies as described in the paper.

#### Use compositionality functions applied to multilingual fastText vectors

Download [fastText word embeddings](https://fasttext.cc/docs/en/crawl-vectors.html) for English, Dutch, French, Spanish, Czech and Japanese. The next step is to align these multilingual word embeddings. Clone fastText:
```bash
git clone https://github.com/facebookresearch/fastText.git
```

Navigate to the `alignment` module, adjust and run the `example.sh` script for each language pair.
In particular, the target language is always set to *en*:
```bash
t=${2:-en}
```
Also the `src_emb` and `tgt_emb` variables should point to the Common Crawl-based vectors previously downloaded, as in the following example:
```bash
src_emb=<download folder>/cc.${s}.300.vec
```

Learn multilingual music genre embeddings:
```bash
cd ccmgp/mgenre_embedding/
python compute_compositional_ft_embeddings.py <folder with aligned fastText vectors>
```

#### Use compositionality functions applied to vectors from transformers' lookup tables
```bash
cd ccmgp/mgenre_embedding/
python compute_compositional_transformer_embeddings.py
```

#### Use contextualized language models as feature extractors
```bash
cd ccmgp/mgenre_embedding/
python compute_transformer_embeddings.py
```

#### Use LASER sentence embedding model
Clone LASER:
```bash
git clone https://github.com/facebookresearch/LASER.git
```

Save in a separate file per language the normalized music genres, which can be obtained as follows:
```python
import ccmgp.utils.utils as utils
from ccmgp.utils.tag_manager import TagManager

genres = utils.get_tags_for_source(lang)
norm_genres = [TagManager.normalize_tag(g, ja=lang == 'ja') for g in genres]
```

Edit and run the `embed.sh` script in the LASER project to compute the embeddings (*INPUT-FILE* contains the normalized music genres in the *LANGUAGE* under consideration):
```bash
cd tasks/embed/
bash ./embed.sh INPUT-FILE LANGUAGE OUTPUT-FILE
```

Transform the raw embeddings in text files:
```python
import argparse
import numpy as np
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sents", required=True)
    parser.add_argument("--embs", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    sents_file = args.sents
    embs_file = args.embs
    out_file = args.out

    with open(sents_file, 'r') as _:
        tags = [l.rstrip() for l in _.readlines()]

    dim = 1024
    X = np.fromfile(embs_file, dtype=np.float32, count=-1)
    X.resize(X.shape[0] // dim, dim)

    df = pd.DataFrame(X, index=tags)
    df.to_csv(out_file, index=True, header=None)
```

Copy the LASER embeddings in a dedicated folder per language, in the `data/laser_embs` folder, under the name *laser.csv*, e.g.:
```bash
mkdir data/laser_embs/en/
cp ../LASER/tasks/embed/en_laser.vec data/laser_embs/en/laser.csv
```

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
