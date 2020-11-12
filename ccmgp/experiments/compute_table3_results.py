import os
import numpy as np

import ccmgp.utils.utils as utils
from ccmgp.utils.judge import Judge
import ccmgp.utils.tag_manager as tagM
from ccmgp.utils.data_helper import DataHelper
from ccmgp.mgenre_embedding.graph_mappers import GraphDirectMapper
from ccmgp.mgenre_embedding.mappers import CompositionEmbsMapper, RetrofitEmbsMapper, LaserEmbsMapper, TransformerEmbsMapper
from ccmgp.mgenre_embedding.googletrans_mapper import GoogleTransMapper


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
        mappers['gtrans'] = GoogleTransMapper(tm, utils.GOOGLETRANS_DIR)
        mappers['dbp_sameas'] = GraphDirectMapper(tm, utils.get_graph())
        mappers['mbert_avg'] = CompositionEmbsMapper(tm, utils.COMP_BERT_EMB_DIR, 'avg')
        mappers['ft_wavg'] = CompositionEmbsMapper(tm, utils.COMP_FT_EMB_DIR, 'wavg')
        mappers['xlm_ctxt'] = TransformerEmbsMapper(tm, os.path.join(utils.TRANSFORM_EMB_DIR, 'xlm_all_token_mean_sent_mean.csv'), 'xlm')
        mappers['laser'] = LaserEmbsMapper(tm, utils.LASER_EMB_DIR, emb_type='laser')
        f = '_'.join(['wavg'] + sorted([source, target]) + ['graph_unaligned_weighted.csv'])
        mappers['rfit_unaligned_ft_wavg'] = RetrofitEmbsMapper(tm, os.path.join(utils.RETRO_EMB_DIR, f), 'wavg')
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
