import os
import numpy as np

import ccmgp.utils.utils as utils
from ccmgp.utils.judge import Judge
import ccmgp.utils.tag_manager as tagM
from ccmgp.utils.data_helper import DataHelper
from ccmgp.mgenre_embedding.mappers import CompositionEmbsMapper, TransformerEmbsMapper


from datetime import datetime
startTime = datetime.now()
for source in utils.langs:
    for target in utils.langs:
        if target == 'en':
            continue
        if source == target:
            target = 'en'
        print(source, '->', target)
        tm = tagM.TagManager.get([source], target)
        datapath = os.path.join(utils.FOLDS_DIR, "{0}_from_{1}_3-fold.tsv".format(target, source))
        print(datapath)
        dhelper = DataHelper(tm, dataset_path=datapath)
        judge = Judge()
        mappers = {}
        for t in utils.emb_composition_types:
            name = 'ft_' + t
            mappers[name] = CompositionEmbsMapper(tm, utils.COMP_FT_EMB_DIR, t)
            name = 'xlm_' + t
            mappers[name] = CompositionEmbsMapper(tm, utils.COMP_XLM_EMB_DIR, t)
            name = 'mbert_' + t
            mappers[name] = CompositionEmbsMapper(tm, utils.COMP_BERT_EMB_DIR, t)
        mappers['xlm_ctxt'] = TransformerEmbsMapper(tm, os.path.join(utils.TRANSFORM_EMB_DIR, 'xlm_all_token_mean_sent_mean.csv'), 'xlm')
        mappers['mbert_ctxt'] = TransformerEmbsMapper(tm, os.path.join(utils.TRANSFORM_EMB_DIR, 'bert_all_token_max_sent_mean.csv'), 'bert')

        for name in mappers:
            print(name)
            mapper = mappers[name]
            results = []
            for fold in range(3):
                eval_data, eval_target = dhelper.get_test_data(fold=fold)
                eval_predicted = mapper.predict_scores(eval_data)
                res = judge.compute_macro_metrics(eval_target, eval_predicted)
                results.append(res)
            print('auc_macro mean ', "%.1f" % (np.mean(results) * 100))
            print('auc_macro std ', "%.1f" % (np.std(results) * 100))
    print(datetime.now() - startTime)
