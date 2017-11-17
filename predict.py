from pymongo import MongoClient

from libtopic_nlp_iit.db_util import *
from libtopic_nlp_iit.data_util import *
from libtopic_nlp_iit.prediction_model import *
from libtopic_nlp_iit.evaluator import *

import sys, os, pickle

def precompute_similarity(db, qf_name, feature_columns):
    sim_mat = {}
    for col in feature_columns:
        sim_path = os.path.join(SERIAL_DIR, "sim_%s_%s"%(qf_name, col))
        if os.path.exists(sim_path): continue
        dbcursor = db[qf_name].find({}, {"web_id": 1, "topic":1, col: 1}, snapshot=True)
        sim_mat["sim_"+col] = get_t2t_sim_matrix(dbcursor2df(dbcursor), col)\
            if not col.startswith("vec") else get_v2v_sim_matrix(dbcursor2df(dbcursor), col)
        pickle.dump(sim_mat["sim_"+col], open(sim_path, 'wb'))

def predict_on_columns(db, qf_name, feature_columns, lbds, ftext=False): #caution: ftext eats up huge memory
    sim_mat = {}
    for col in feature_columns:
        sim_path = os.path.join(SERIAL_DIR, "sim_%s_%s"%(qf_name, col))
        if os.path.exists(sim_path):
            sim_mat["sim_"+col] = pickle.load(open(sim_path, 'rb'))
        # prediction dict for all documents of a query focus under all thresholds lbds
        preds = None
        if col.startswith("vec"):
            preds = UnsupervisedPredictor(sim_mat, db[qf_name]).predict_subtopics_on_vec(col, lbds)
        else:
            preds = UnsupervisedPredictor(sim_mat, db[qf_name]).predict_subtopics_on_bow(col, lbds)
        dict_attrib = {"_id": 0, "web_id": 1, "topic": 1, "url": 1, "suggest_f": 1, "label": 1, "topic_probability": 1}
        if ftext: dict_attrib["content"] = 1
        df_col = dbcursor2df(db[qf_name].find({}, dict_attrib, snapshot=True))
        for lbd, pred in zip(lbds, preds):
            #print(col, lbd, len(pred))
            df_col["%.2f"%lbd] = df_col["web_id"].apply(lambda wid: pred[wid])
        try:
            os.stat(OUT_DIR)
        except OSError:
            os.mkdir(OUT_DIR)
        df_col.to_csv(os.path.join(OUT_DIR,"pred_%s_%s.csv"%(qf_name, col)), index=False)

def predict_on_rank(db, qf_name, rlbds, ftext=False): #caution: ftext eats up huge memory
    preds = UnsupervisedPredictor({}, db[qf_name]).predict_subtopics_on_topic_ranking(rlbds)
    dict_attrib = {"_id": 0, "web_id": 1, "topic": 1, "url": 1, "suggest_f": 1, "label": 1, "topic_probability": 1}
    if ftext: dict_attrib["content"] = 1
    df_col = dbcursor2df(db[qf_name].find({}, dict_attrib, snapshot=True))
    for rlbd, pred in zip(rlbds, preds):
        #print(col, lbd, len(pred))
        df_col["%d"%rlbd] = df_col["web_id"].apply(lambda wid: pred[wid])
    try:
        os.stat(OUT_DIR)
    except OSError:
        os.mkdir(OUT_DIR)
    df_col.to_csv(os.path.join(OUT_DIR,"pred_%s_rank.csv"%qf_name), index=False)

def main(argv):
    global SERIAL_DIR; SERIAL_DIR = os.path.join(os.getcwd(), "serialization")
    global OUT_DIR; OUT_DIR = os.path.join(os.getcwd(), "output")
    global qf_name_dict; qf_name_dict = dict([("kafunsyo", "花粉症"),
                                              ("kekkon", "結婚"),
                                              ("syukatsu", "就活"),
                                              ("3dprinter", "3dプリンタ"),
                                              ("mansion", "マンション")])
    
    client = MongoClient('localhost', 27017)#['test-database']
    db = client["dataset"]
    qf_name = argv[0]
    collection = db[qf_name]
    feature_columns = ['token', 'vec_doc', 'vec_sKeyWord_5', 'vec_sKeyWord_8', 'vec_sKeyWord_10', 'vec_sKeyWord_wikiOnly_5', 'vec_sKeyWord_wikiOnly_8', 'vec_sKeyWord_wikiOnly_10']
    precompute_similarity(db, qf_name, feature_columns)
    predict_on_columns(db, qf_name, feature_columns, np.arange(-1, 1.01, 0.02))
    predict_on_rank(db, qf_name, list(range(0, 31)))

if __name__ == "__main__":
    main(sys.argv[1:])









