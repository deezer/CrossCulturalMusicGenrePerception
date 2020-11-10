import os
import sys
import torch
from transformers import XLMModel, XLMTokenizer, BertTokenizer, BertModel
import pandas as pd

import ccmgp.utils.tag_manager as tm
import ccmgp.utils.utils as utils


class TransformerEmbeddingGenerator:
    """Tag embedding generator from using pre-trained contextual models based on transformers"""

    def __init__(self, model_type):
        """Constructor
        :param model_type: if and xlm or bert model is used
        """
        # Instantiate model and tokenizers from pre-trained multilingual versions
        if model_type == 'xlm':
            self.tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-100-1280')
            self.model = XLMModel.from_pretrained('xlm-mlm-100-1280', output_hidden_states=True)
        elif model_type == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
            self.model = BertModel.from_pretrained('bert-base-multilingual-uncased', output_hidden_states=True)
        else:
            raise ValueError('Unrecognized model type. Only bert and xlm supported')

    def compute_embs(self, tags):
        """Compute contextual embeddings for the given tags
        """
        def get_token_embeddings(all_layers, pool_f=torch.mean, layers=[-1]):
            """ Compute embedding for a token with different pooling strategies over all layers
            """
            token_vecs = token_vecs = torch.stack([all_layers[i] for i in layers], dim=0)
            token_vecs = torch.squeeze(token_vecs, dim=1)
            token_embs = pool_f(token_vecs, dim=0)
            if pool_f == torch.max:
                token_embs = token_embs[0]
            return token_embs

        def get_sent_embedding(token_vecs, pool_f=torch.mean):
            """ Compute embedding for a sentence / in this case multi-word expressions
            To ensure a fixed-length representation averaging token embeddings is used by default
            """
            sent_emb = pool_f(token_vecs, dim=0)
            if pool_f == torch.max:
                sent_emb = sent_emb[0]
            return sent_emb.tolist()

        results = {'last_sent_mean': {}, 'last_sent_max': {}, 'second_to_last_sent_mean': {}, 'second_to_last_sent_max': {}, 'last_four_token_mean_sent_mean': {}, 'last_four_token_mean_sent_max': {}, 'last_four_token_max_sent_mean': {}, 'last_four_token_max_sent_max': {}, 'all_token_mean_sent_mean': {}, 'all_token_mean_sent_max': {}, 'all_token_max_sent_mean': {}, 'all_token_max_sent_max': {}}
        for t in set(tags.values()):
            with torch.no_grad():
                input_ids = torch.tensor(self.tokenizer.encode(t)).unsqueeze(0)
                model_outputs = self.model(input_ids)
                all_layers = model_outputs[-1]

                # Multiple strategies are explored
                # 1. Pool last layer
                token_vecs = get_token_embeddings(all_layers, layers=[-1])
                results['last_sent_mean'][t] = get_sent_embedding(token_vecs, torch.mean)
                results['last_sent_max'][t] = get_sent_embedding(token_vecs, torch.max)

                # 2. Pool second to last layer
                token_vecs = get_token_embeddings(all_layers, layers=[-2])
                results['second_to_last_sent_mean'][t] = get_sent_embedding(token_vecs, torch.mean)
                results['second_to_last_sent_max'][t] = get_sent_embedding(token_vecs, torch.max)

                # 3. Pool the last 4 layers
                token_vecs_mean = get_token_embeddings(all_layers, pool_f=torch.mean, layers=list(range(-4, 0, 1)))
                token_vecs_max = get_token_embeddings(all_layers, pool_f=torch.max, layers=list(range(-4, 0, 1)))
                results['last_four_token_mean_sent_mean'][t] = get_sent_embedding(token_vecs_mean, torch.mean)
                results['last_four_token_mean_sent_max'][t] = get_sent_embedding(token_vecs_mean, torch.max)
                results['last_four_token_max_sent_mean'][t] = get_sent_embedding(token_vecs_max, torch.mean)
                results['last_four_token_max_sent_max'][t] = get_sent_embedding(token_vecs_max, torch.max)

                # 4. Pool all layers (all_layers contains embeddings too in the first position)
                token_vecs_mean = get_token_embeddings(all_layers, pool_f=torch.mean, layers=list(range(1, len(all_layers))))
                token_vecs_max = get_token_embeddings(all_layers, pool_f=torch.max, layers=list(range(1, len(all_layers))))
                results['all_token_mean_sent_mean'][t] = get_sent_embedding(token_vecs_mean, torch.mean)
                results['all_token_mean_sent_max'][t] = get_sent_embedding(token_vecs_mean, torch.max)
                results['all_token_max_sent_mean'][t] = get_sent_embedding(token_vecs_max, torch.mean)
                results['all_token_max_sent_max'][t] = get_sent_embedding(token_vecs_max, torch.max)
        return results


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: compute_transformer_embeddings.py [xlm|bert]")
        sys.exit(1)
    model_type = sys.argv[1]
    output_embs_dir = utils.TRANSFORM_EMB_DIR
    if not os.path.exists(output_embs_dir):
        os.makedirs(output_embs_dir)

    tags = {}
    for lang in utils.langs:
        # Retrieve tags per language
        genres = utils.get_tags_for_source(lang)
        # Normalize tags
        norm_genres = [tm.TagManager.normalize_tag(g, ja=lang == 'ja') for g in genres]
        # Keep track of the prefix formed too which contains info about lang
        prefix_norm_genres = [lang + ':' + ng for ng in norm_genres]
        tags.update(zip(prefix_norm_genres, norm_genres))
    # Instantiate the generator for contextual embeddings
    emb_generator = TransformerEmbeddingGenerator(model_type)
    # Compute and save embeddings
    results = emb_generator.compute_embs(tags)
    for emb_type in results:
        embs = {}
        for t in tags:
            embs[t] = results[emb_type][tags[t]]
        df = pd.DataFrame(embs.values(), index=embs.keys())
        df.to_csv(os.path.join(output_embs_dir, model_type + '_' + emb_type + '.csv'), header=False)

