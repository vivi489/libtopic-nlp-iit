# -*- coding: utf-8 -*-
import graphlab, os, pickle, sys, re
from collections import defaultdict, Counter
from scipy.spatial.distance import cosine
from operator import itemgetter
import numpy as np
from data_util import *
from unsupervised_models import *


SERIAL_DIR = os.path.join(os.getcwd(), "serialization")
WORKING_DATA_FRAME_DIR = os.path.join(os.getcwd(), "working_data")
qf_name_dict = dict([("kafunsyo", "花粉症"), ("kekkon", "結婚"), ("syukatsu", "就活"), ("3dprinter", "3dプリンタ"), ("mansion", "マンション")])

qf_name, sf = sys.argv[1], graphlab.SFrame(os.path.join(WORKING_DATA_FRAME_DIR, sys.argv[1]))
window_size = int(sys.argv[2])
micro = sys.argv[3]=="micro"

if not len(sys.argv) == 4:
    print "invalid argument list"
    exit(0)

set_conf = set([line.strip() for line in open("predict.conf", 'r')])

def predict(qf_name, sf, window_size, micro=True):
    d2d_vsim = pickle.load(open(os.path.join(SERIAL_DIR, "d2d_vsim_%s"%qf_name)))
    d2d_wsim = pickle.load(open(os.path.join(SERIAL_DIR, "d2d_wsim_%s"%qf_name)))
    predictor = UnsupervisedPredictor(d2d_vsim=d2d_vsim, d2d_wsim=d2d_wsim, list_sframe_topic=DataUtility.make_list_by_topic(sf))
    predictor.qf_name = qf_name
    predictor.qf_name_JPC = qf_name_dict[qf_name]
    sa_list_label = [sf_t["label"] for sf_t in predictor.list_sframe_topic]
    eval_stat = defaultdict(list)
    
    for lbd in np.arange(-1.0, 1.02, 0.02):
        #print qf_name, "lbd =", lbd
        #eval_stat["lbd"].append(lbd)
        if "predict_subtopics_on_vec_doc" in set_conf:
            sa_list_prediction_doc = predictor.predict_subtopics_on_vec_doc(SIM_LB=lbd)
            evaluator = Evaluator(sa_list_prediction=sa_list_prediction_doc, sa_list_label=sa_list_label)
            #eval_stat["accuracy_doc"].append(evaluator.get_accuracy())
            p_r = evaluator.get_micro_pre_rec() if micro else evaluator.get_macro_pre_rec()
            eval_stat["p_doc"].append(p_r[0])
            eval_stat["r_doc"].append(p_r[1])
            del evaluator
            del sa_list_prediction_doc
        
        if "predict_subtopics_on_sKeyWord" in set_conf:
            sa_list_prediction_sKeyWord = predictor.predict_subtopics_on_sKeyWord_vec("vec_sKeyWord_%d"%window_size, SIM_LB=lbd)
            evaluator = Evaluator(sa_list_prediction=sa_list_prediction_sKeyWord, sa_list_label=sa_list_label)
            #eval_stat["accuracy_sKeyWord"].append(evaluator.get_accuracy())
            p_r = evaluator.get_micro_pre_rec() if micro else evaluator.get_macro_pre_rec()
            eval_stat["p_sKeyWord"].append(p_r[0])
            eval_stat["r_sKeyWord"].append(p_r[1])
            del evaluator
            del sa_list_prediction_sKeyWord
        
        if "predict_subtopics_on_sKeyWord_wikiOnly" in set_conf:
            sa_list_prediction_sKeyWord_wikiOnly = predictor.predict_subtopics_on_sKeyWord_vec("vec_sKeyWord_wikiOnly_%d"%window_size, SIM_LB=lbd)
            evaluator = Evaluator(sa_list_prediction=sa_list_prediction_sKeyWord_wikiOnly, sa_list_label=sa_list_label)
            #eval_stat["accuracy_sKeyWord_wikiOnly"].append(evaluator.get_accuracy())
            p_r = evaluator.get_micro_pre_rec() if micro else evaluator.get_macro_pre_rec()
            eval_stat["p_sKeyWord_wikiOnly"].append(p_r[0])
            eval_stat["r_sKeyWord_wikiOnly"].append(p_r[1])
            del evaluator
            del sa_list_prediction_sKeyWord_wikiOnly
        
    if "predict_subtopics_on_bow" in set_conf:
        for lbd in np.arange(0.0, 1.02, 0.02):
            sa_list_prediction_bow = predictor.predict_subtopics_on_bow(SIM_LB=lbd)
            evaluator = Evaluator(sa_list_prediction=sa_list_prediction_bow, sa_list_label=sa_list_label)
            #eval_stat["accuracy_sKeyWord_bow"].append(evaluator.get_accuracy())
            p_r = evaluator.get_micro_pre_rec() if micro else evaluator.get_macro_pre_rec()
            eval_stat["p_bow"].append(p_r[0])
            eval_stat["r_bow"].append(p_r[1])
            del evaluator
            del sa_list_prediction_bow
        max_len = max([len(l) for l in eval_stat.values()])
        cur_len = len(eval_stat["p_bow"])
        eval_stat["p_bow"].extend([np.NaN]*(max_len-cur_len))
        eval_stat["r_bow"].extend([np.NaN]*(max_len-cur_len))

    if "predict_subtopics_on_topic_ranking" in set_conf:
        for lbd in range(0, 31):
            #print qf_name, "lbd =", lbd
            sa_list_prediction_ranking = predictor.predict_subtopics_on_topic_ranking(RANK_LB=lbd)
            evaluator = Evaluator(sa_list_prediction=sa_list_prediction_ranking, sa_list_label=sa_list_label)
            p_r = evaluator.get_micro_pre_rec() if micro else evaluator.get_macro_pre_rec()
            eval_stat["p_ranking"].append(p_r[0])
            eval_stat["r_ranking"].append(p_r[1])
            del evaluator
            del sa_list_prediction_ranking
        max_len = max([len(l) for l in eval_stat.values()])
        cur_len = len(eval_stat["p_ranking"])
        eval_stat["p_ranking"].extend([np.NaN]*(max_len-cur_len))
        eval_stat["r_ranking"].extend([np.NaN]*(max_len-cur_len))

    #graphlab.SFrame(eval_stat)["p_doc", "r_doc", "p_sKeyWord", "r_sKeyWord", "p_sKeyWord_wikiOnly", "r_sKeyWord_wikiOnly",\
    #"p_sKeyWord_bow", "r_sKeyWord_bow", "p_suggests", "r_suggests"].save("stat_%s_micro_%d.csv"%(qf_name, window_size), format="csv")
    graphlab.SFrame(eval_stat).save("stat_%s_%s_%d.csv"%(qf_name, sys.argv[3], window_size), format="csv")
    
    del predictor

if __name__ == "__main__":
    predict(qf_name, sf, window_size, micro)
