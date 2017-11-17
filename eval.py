import pandas as pd
from libtopic_nlp_iit.evaluator import *

from collections import defaultdict
import pandas as pd
import numpy as np
import os, sys

def split_df_by_topic(df):
    retVal = []
    p0, p1 = 0, 0
    while p1 < len(df):
        while p1 < len(df) and df["topic"][p1]==df["topic"][p0]:
            p1 += 1
        if p0 < p1:
            retVal.append(df[p0: p1])
        p0 = p1
    return retVal

def eval_csv(qf_name, column_name):
    path = os.path.join(OUT_DIR, "pred_%s_%s.csv"%(qf_name, column_name))
    df = pd.read_csv(path)
    df.sort_values("topic", inplace=True)
    list_df_t = split_df_by_topic(df)
    lists_label = [df_t["label"] for df_t in list_df_t]
    retVal = defaultdict(list)
    for lbd in np.arange(-1, 1.01, 0.02):
        lists_prediction = [df_t["%.2f"%lbd] for df_t in list_df_t]
        evaluator = Evaluator(lists_prediction, lists_label)
        macro_pre, macro_re = evaluator.get_macro_pre_rec()
        micro_pre, micro_re = evaluator.get_micro_pre_rec()
        #print(macro_pre, macro_re, micro_pre, micro_re)
        retVal["lbd"].append(lbd)
        retVal["macro_pre"].append(macro_pre)
        retVal["macro_re"].append(macro_re)
        retVal["micro_pre"].append(micro_pre)
        retVal["micro_re"].append(micro_re)
    return pd.DataFrame(retVal, columns=["lbd", "macro_re", "macro_pre", "micro_re", "micro_pre"])

def eval_csv_rank(qf_name, column_name):
    path = os.path.join(OUT_DIR, "pred_%s_%s.csv"%(qf_name, column_name))
    df = pd.read_csv(path)
    df.sort_values("topic", inplace=True)
    list_df_t = split_df_by_topic(df)
    lists_label = [df_t["label"] for df_t in list_df_t]
    retVal = defaultdict(list)
    for rlbd in range(30):
        lists_prediction = [df_t["%d"%rlbd] for df_t in list_df_t]
        evaluator = Evaluator(lists_prediction, lists_label)
        macro_pre, macro_re = evaluator.get_macro_pre_rec()
        micro_pre, micro_re = evaluator.get_micro_pre_rec()
        #print(macro_pre, macro_re, micro_pre, micro_re)
        retVal["rlbd"].append(rlbd)
        retVal["macro_pre"].append(macro_pre)
        retVal["macro_re"].append(macro_re)
        retVal["micro_pre"].append(micro_pre)
        retVal["micro_re"].append(micro_re)
    return pd.DataFrame(retVal, columns=["rlbd", "macro_re", "macro_pre", "micro_re", "micro_pre"])

def main(argv):
    global OUT_DIR; OUT_DIR = os.path.join(os.getcwd(), "output")
    feature_columns = ['token', 'vec_doc', 'vec_sKeyWord_5', 'vec_sKeyWord_8', 'vec_sKeyWord_10', 'vec_sKeyWord_wikiOnly_5', 'vec_sKeyWord_wikiOnly_8', 'vec_sKeyWord_wikiOnly_10']
    qf_name = argv[0]
    for column_name in feature_columns:
        eval_csv(qf_name, column_name).to_csv(os.path.join(OUT_DIR, "eval_%s_%s.csv"%(qf_name, column_name)))
    eval_csv_rank(qf_name, "rank").to_csv(os.path.join(OUT_DIR, "eval_%s_%s.csv"%(qf_name, "rank")))

if __name__ == "__main__":
    main(sys.argv[1:])
