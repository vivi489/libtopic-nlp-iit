import os, pickle, sys, re
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec
from scipy.spatial.distance import cosine
from collections import defaultdict, Counter
from operator import itemgetter

from math import sqrt

def get_suggest_f_tuple_map(df): #pandas frame (web_id, suggests, topic)
    #return a dict: web_id -> suggest_f tuple list
    retVal = {}
    for t in sorted(list(set(df["topic"]))):
        suggest_f_dict = defaultdict(int)
        df_t = df[df["topic"]==t]
        for suggests in df_t["suggests"]:
            for suggest in suggests.split('\n'):
                if len(suggest) == 0: continue
                suggest_f_dict[suggest] += 1
        for doc in df_t.iterrows(): #doc is tuple of (int, Series)
            suggest_f = [(s, suggest_f_dict[s]) for s in doc[1]["suggests"].split("\n") if len(s)>0]
            suggest_f.sort(key=itemgetter(1), reverse=True)
            retVal[doc[1]["web_id"]] = suggest_f
    #print("topic %d: %d"%(t, len(suggest_f_array)))
    return retVal

def get_dict_suggest_word_vec(set_valid_suggest_words, w2v_model_path):
    model = Word2Vec.load(w2v_model_path)
    dict_wiki_w2v = {}
    for w in model.wv.vocab:
        if w in set_valid_suggest_words:
            dict_wiki_w2v[w] = model[w]
    return dict_wiki_w2v


def get_doc_vec_list(d2v_model_path):
    model = Doc2Vec.load(d2v_model_path)
    return list(model.docvecs)


def get_valid_sKeyWord_vec(doc, qf_name_JPC, suggest_word_vec):
    for s_and_f in doc["suggest_f"]:
        suggest_tokens = set(re.split("\s+", s_and_f[0]))
        if qf_name_JPC in suggest_tokens:
            suggest_tokens.remove(qf_name_JPC)
        if (not len(suggest_tokens) == 1) or (not list(suggest_tokens)[0] in suggest_word_vec):
            continue
        return suggest_word_vec[list(suggest_tokens)[0]]
    return []

def get_valid_sKeyWord(doc, qf_name_JPC, suggest_word_vec):
    for s_and_f in doc["suggest_f"]:
        suggest_tokens = set(re.split("\s+", s_and_f[0]))
        if qf_name_JPC in suggest_tokens:
            suggest_tokens.remove(qf_name_JPC)
        if (not len(suggest_tokens) == 1) or (not list(suggest_tokens)[0] in suggest_word_vec):
            continue
        return list(suggest_tokens)[0]
    return ""

def get_v2v_sim_matrix(df, attrib_name): #df can be a dataframe of multiple topics; attrib_name column is required
    d2d_sim = {}
    df[attrib_name] = df[attrib_name].apply(lambda x: pickle.loads(x))
    df = df[df[attrib_name].apply(lambda x: len(x)>0)]
    for t in set(df["topic"]):
        df_t = df[df["topic"]==t]
        for i in range(len(df_t)):
            wid_i = df_t.iloc[i]["web_id"]
            for j in range(i+1, len(df_t)):
                wid_j = df_t.iloc[j]["web_id"]
                if wid_i < wid_j:
                    d2d_sim[(wid_i, wid_j)] = 1 - cosine(df_t.iloc[i][attrib_name], df_t.iloc[j][attrib_name])
                else:
                    d2d_sim[(wid_j, wid_i)] = 1 - cosine(df_t.iloc[i][attrib_name], df_t.iloc[j][attrib_name])
    return d2d_sim

def get_t2t_sim_matrix(df, attrib_name):
    t2t_sim = {}
    df[attrib_name] = df[attrib_name].apply(lambda x: re.split("\s+", x.strip()))
    df[attrib_name] = df[attrib_name].apply(lambda x: Counter(x))
    for t in set(df["topic"]):
        df_t = df[df["topic"]==t]
        for i in range(len(df_t)):
            wid_i = df_t.iloc[i]["web_id"]
            for j in range(i+1, len(df_t)):
                wid_j = df_t.iloc[j]["web_id"]
                if wid_i < wid_j:
                    t2t_sim[(wid_i, wid_j)] = dict_cosine(df_t.iloc[i][attrib_name], df_t.iloc[j][attrib_name])
                else:
                    t2t_sim[(wid_j, wid_i)] = dict_cosine(df_t.iloc[i][attrib_name], df_t.iloc[j][attrib_name])
    return t2t_sim

def dict_cosine(doc1, doc2): #dict-like inputs
    numerator = 0
    norm1 = 0
    for k1, v1 in doc1.items():
        numerator += v1 * doc2.get(k1, 0.0)
        norm1 += v1 * v1
    norm2 = sum([v2*v2 for v2 in doc2.values()])
    return numerator / sqrt(norm1 * norm2)






