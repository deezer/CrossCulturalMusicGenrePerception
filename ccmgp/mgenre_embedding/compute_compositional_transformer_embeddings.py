import os

import ccmgp.utils.tag_manager as tm
import ccmgp.utils.utils as utils

from composition_emb_generators import TransformerEmbeddingGenerator

out_dir = utils.COMP_FT_EMB_DIR
langs = utils.langs
emb_composition_types = utils.emb_composition_types
for model_type in ['xlm', 'bert']:
    emb_generator = TransformerEmbeddingGenerator(model_type)
    for lang in utils.langs:
        print(lang)
        lang_embs_out = os.path.join(out_dir, model_type, lang)
        if not os.path.exists(lang_embs_out):
            os.makedirs(lang_embs_out)
        #print(emb_generator.emb_dim)
        # Retrieve normalized tags
        norm_tags = set([tm.TagManager.normalize_tag(t, ja=lang=='ja') for t in utils.get_tags_for_source(lang)])
        # And the original tag with a basic normalization
        orig_tags = set([tm.TagManager.normalize_tag_basic(t, prefixed=True) for t in utils.get_tags_for_source(lang)])
        # Generate embeddings and save them
        for ect in emb_composition_types:
            lang_out_path = os.path.join(lang_embs_out, ect + '.csv')
            if ect == 'avg':
                emb_generator.get_avg_embs(norm_tags, lang_out_path)
            elif ect == 'wavg':
                emb_generator.get_wei_avg_embs(norm_tags, save_file_path=lang_out_path)
            else:
                print("Embedding type not supported")
