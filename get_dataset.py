from libtopic_nlp_iit.db_util import *
from libtopic_nlp_iit.data_util import *
from pymongo import MongoClient
import sys, os, pickle


def main(argv):
    global SERIAL_DIR; SERIAL_DIR = os.path.join(os.getcwd(), "serialization")
    global W2V_DIR; W2V_DIR = os.path.join(os.path.dirname(os.getcwd()), "journal_related_new_data", "word2vec")
    global D2V_DIR; D2V_DIR = os.path.join(os.path.dirname(os.getcwd()), "journal_related_new_data", "doc2vec")
  
    global qf_name_dict; qf_name_dict = dict([("kafunsyo", "花粉症"),
                                             ("kekkon", "結婚"),
                                             ("syukatsu", "就活"),
                                             ("3dprinter", "3dプリンタ"),
                                             ("mansion", "マンション")])
    qf_names = ["syukatsu", "kekkon", "kafunsyo", "mansion"]

    try:
        os.stat(SERIAL_DIR)
    except OSError:
        os.mkdir(SERIAL_DIR)

    client = MongoClient('localhost', 27017)
    db = client["query_focus"]
    client.drop_database("dataset")
    db_dataset = client["dataset"]
    for qf_name in qf_names:
        make_dataset(db, qf_name, db_dataset, tokenize=True)


def serialize(db, qf_name):
    print("%s: serialization starts"%qf_name)
    if not os.path.exists(os.path.join(SERIAL_DIR, "doc2vec_model_%s"%qf_name)):
        doc_vec_list = get_doc_vec_list(os.path.join(D2V_DIR, "doc2vec_model_%s"%qf_name)) #numpy array
        pickle.dump(doc_vec_list, open(os.path.join(SERIAL_DIR, "vec_doc_%s"%qf_name), 'wb'))
    if not os.path.exists(os.path.join(SERIAL_DIR, "set_valid_suggest_words_%s"%qf_name)):
        set_valid_suggest_words = extract_one_segment_suggest_keywords(db, qf_name, qf_name_dict[qf_name])
        pickle.dump(set_valid_suggest_words,
                    open(os.path.join(SERIAL_DIR, "set_valid_suggest_words_%s"%qf_name), 'wb'))
    for size in [5, 8, 10]:
        if not os.path.exists(os.path.join(SERIAL_DIR, "suggest_word_vec_%s_size_%d"%(qf_name, size))):
            dict_wiki_w2v = get_dict_suggest_word_vec(set_valid_suggest_words,\
                                                      os.path.join(W2V_DIR, "trained_models", "model_%d_%s.model"%(size, qf_name)))
            pickle.dump(dict_wiki_w2v,\
                        open(os.path.join(SERIAL_DIR, "suggest_word_vec_%s_size_%d"%(qf_name, size)), 'wb'))
        if not os.path.exists(os.path.join(SERIAL_DIR, "suggest_word_vec_no_%s_size_%d"%(qf_name, size))):
            dict_wiki_w2v = get_dict_suggest_word_vec(set_valid_suggest_words,\
                                                      os.path.join(W2V_DIR, "trained_models", "model_%d_no_%s.model"%(size, qf_name)))
            pickle.dump(dict_wiki_w2v,\
                        open(os.path.join(SERIAL_DIR, "suggest_word_vec_no_%s_size_%d"%(qf_name, size)), 'wb'))
    print ("%s: done with serialization"%qf_name)


def make_dataset(db, qf_name, db_dataset, tokenize=False):
    serialize(db, qf_name)
    dbcursor = db[qf_name].find({}, {"web_id": 1, "topic":1, "suggests":1}, snapshot=True)
    dict_suggest_f = get_suggest_f_tuple_map(dbcursor2df(dbcursor)) #map: web_id -> tuple list
    df_target = pd.read_csv("label_%s.csv"%qf_name)
    df_target = df_target.loc[[~df_target["label"].isnull()][0]]
    
    # docs were sorted on (topic, topic_probability) at doc2vec training
    # the next line assumes wrong order
    #list_vec_doc = get_doc_vec_list(os.path.join(D2V_DIR, "doc2vec_model_%s"%qf_name))
    topics = dbcursor.distinct(key="topic")
    doc_count = 0
    index2docvecpos = {}
    for t in sorted(topics):
        dbcursor = db[qf_name].find({"topic": t}, {"web_id": 1, "index": 1})\
            .sort([("topic_probability", pymongo.DESCENDING)])
        for doc in dbcursor:
            index2docvecpos[doc["index"]] = doc_count
            doc_count += 1
    #print("doc_count=", doc_count)
    list_vec_doc = get_doc_vec_list(os.path.join(D2V_DIR, "doc2vec_model_%s"%qf_name))
    #print("list_vec_doc size=", len(list_vec_doc))
    list_window_size = [5, 8, 10]
    list_suggest_word_vec, list_suggest_word_vec_no = [], []
    for size in list_window_size:
        list_suggest_word_vec.append(pickle.load(open(os.path.join(SERIAL_DIR, "suggest_word_vec_%s_size_%d"%(qf_name, size)), "rb")))
        list_suggest_word_vec_no.append(pickle.load(open(os.path.join(SERIAL_DIR, "suggest_word_vec_no_%s_size_%d"%(qf_name, size)), "rb")))
    for row in df_target.iterrows():
        wid = int(row[1]["web_id"])
        doc = db[qf_name].find_one({"web_id": wid}, {"_id": False})
        doc["label"] = row[1]["label"]
        doc["suggest_f"] = dict_suggest_f[doc["web_id"]]
        doc["vec_doc"] = pickle.dumps(list_vec_doc[index2docvecpos[doc["index"]]])
        for size,  suggest_word_vec, suggest_word_vec_no in\
            zip(list_window_size, list_suggest_word_vec, list_suggest_word_vec_no):
            doc["sKeyWord_%d"%size] = get_valid_sKeyWord(doc, qf_name_dict[qf_name], suggest_word_vec)
            doc["vec_sKeyWord_%d"%size] = pickle.dumps(get_valid_sKeyWord_vec(doc, qf_name_dict[qf_name], suggest_word_vec))
            doc["sKeyWord_wikiOnly_%d"%size] = get_valid_sKeyWord(doc, qf_name_dict[qf_name], suggest_word_vec_no)
            doc["vec_sKeyWord_wikiOnly_%d"%size] = pickle.dumps(get_valid_sKeyWord_vec(doc, qf_name_dict[qf_name], suggest_word_vec_no))
        insert_doc(db_dataset, "%s"%qf_name, doc)
    if tokenize:
        print("%s: tokenizing.."%qf_name)
        tokenize_doc_collection(db_dataset, qf_name)
    print("%s: completed\n"%qf_name)

if __name__ == "__main__":
    main(sys.argv[1:])



