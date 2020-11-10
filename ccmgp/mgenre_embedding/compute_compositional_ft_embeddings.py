import os
import sys

import ccmgp.utils.tag_manager as tm
import ccmgp.utils.utils as utils

from composition_emb_generators import FTEmbeddingGenerator

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: compute_compositional_ft_embeddings.py ft_aligned_emb_dir")
        sys.exit(1)
    aligned_emb_dir = sys.argv[1]
    out_dir = utils.COMP_FT_EMB_DIR
    langs = utils.langs
    emb_composition_types = utils.emb_composition_types

    for lang in utils.langs:
        print(lang)
        lang_embs_path = os.path.join(aligned_emb_dir, ''.join(['cc.', lang, '-en.vec']))
        lang_embs_out = os.path.join(out_dir, lang)
        if not os.path.exists(lang_embs_out):
            os.makedirs(lang_embs_out)
        ft_emb_generator = FTEmbeddingGenerator(lang_embs_path)
        #print("Emb dim", ft_emb_generator.emb_dim)
        # Retrieve normalized tags
        norm_tags = set([tm.TagManager.normalize_tag(t, ja=lang=='ja') for t in utils.get_tags_for_source(lang)])
        # And the original tag with a basic normalization
        orig_tags = set([tm.TagManager.normalize_tag_basic(t, prefixed=True) for t in utils.get_tags_for_source(lang)])
        # Generate embeddings and save them
        for ect in emb_composition_types:
            lang_out_path = os.path.join(out_dir, lang, ect + '.csv')
            if ect == 'avg':
                ft_emb_generator.get_avg_embs(norm_tags, lang_out_path)
            elif ect == 'wavg':
                ft_emb_generator.get_wei_avg_embs(norm_tags, save_file_path=lang_out_path)
            else:
                print("Embedding type not supported")
