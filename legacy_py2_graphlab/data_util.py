# -*- coding: utf-8 -*-
from scipy.sparse import *
import graphlab, os, pickle, sys, re, gensim
from gensim.models import Word2Vec
from collections import defaultdict, Counter
from scipy.spatial.distance import cosine
from operator import itemgetter
from gensim.models.doc2vec import Doc2Vec

class DataUtility: #helper class for utility methods
    @staticmethod
    def make_list_by_topic(sf):
        retVal = []
        for t in set(sf["topic"]):
            retVal.append(sf[sf["topic"]==t])
        return retVal
    
    @staticmethod
    def get_topN(sf_qf, N):
        retVal = None
        for t in xrange(sf_qf["topic"].max()+1):
            sf_t = sf_qf[sf_qf["topic"]==t][:N]
            retVal = sf_t if retVal is None else retVal.append(sf_t)
        return retVal
    
    @staticmethod
    def get_d2d_vsim_matrix(sf): #sf can be doc sframe of multiple topics; "vec_doc" column is required
        d2d_vsim = {}
        for t in set(sf["topic"]):
            sf_t = sf[sf["topic"]==t]
            for i in range(len(sf_t)):
                for j in range(len(sf_t)):
                    if sf_t[i]["web_id"] < sf_t[j]["web_id"]:
                        d2d_vsim[(sf_t[i]["web_id"], sf_t[j]["web_id"])] = 1 - cosine(sf_t[i]["vec_doc"], sf_t[j]["vec_doc"])
                    else:
                        d2d_vsim[(sf_t[j]["web_id"], sf_t[i]["web_id"])] = 1 - cosine(sf_t[i]["vec_doc"], sf_t[j]["vec_doc"])
        return d2d_vsim
    
    @staticmethod
    def get_d2d_sim_matrix(sf, token_col, n=1): #sf can be a doc sframe of multiple topics; token column needs specifying
        d2d_sim = {}
        sf["ngram"] = graphlab.text_analytics.count_ngrams(sf[token_col], n);
        for t in set(sf["topic"]):
            sf_t = sf[sf["topic"]==t]
            for i in range(len(sf_t)):
                #token_i = dict(Counter(re.split("\s+", sf_t[i]["token"])))
                for j in range(len(sf_t)):
                    #token_j = dict(Counter(re.split("\s+", sf_t[j]["token"])))
                    if sf_t[i]["web_id"] < sf_t[j]["web_id"]:
                        d2d_sim[(sf_t[i]["web_id"], sf_t[j]["web_id"])] = 1 - graphlab.toolkits.distances.cosine(sf_t[i]["ngram"], sf_t[j]["ngram"])
                    else:
                        d2d_sim[(sf_t[j]["web_id"], sf_t[i]["web_id"])] = 1 - graphlab.toolkits.distances.cosine(sf_t[i]["ngram"], sf_t[j]["ngram"])
        sf.remove_column("ngram")
        return d2d_sim
    
    @staticmethod
    def create_labeled_doc_set(sf, sf_label): #needs pre-converted sframe from csv
        sf_label = sf_label[sf_label["label"].apply(lambda x: len(x)>0)]
        return sf.join(sf_label[["web_id", "label"]], on=["web_id"], how="right")
    
    @staticmethod
    def get_suggest_f_col(sf):
        #return an SArray of lists of (s, f) tuples
        retVal = []
        #dict_sugg_f_list = pickle.load(open(os.path.join(os.path.dirname(os.getcwd()), "serialization", "dict_sugg_f_list_syukatsu"), 'r'))
        for t in sorted(list(set(sf["topic"]))):
            suggest_f_dict = defaultdict(int)
            sf_t = sf.filter_by(t, "topic")
            suggest_f_array = []
            for suggests in sf_t["suggests"]:
                for suggest in suggests.split('\n'):
                    if len(suggest) == 0: continue
                    suggest_f_dict[suggest] += 1
            for doc in sf_t:
                suggest_f = [(s, suggest_f_dict[s]) for s in doc["suggests"].split("\n") if len(s)>0]
                suggest_f.sort(key=itemgetter(1), reverse=True)
                suggest_f_array.append(suggest_f)
            retVal.extend(suggest_f_array)
        return graphlab.SArray(retVal)
    
    @staticmethod
    def extract_one_segment_suggest_keywords(sf, qf_name_JPC): #used to retrieve set_valid_suggest_words for a whole query focus
        #return a set of keywords
        retVal = set()
        for suggests in sf["suggests"]:
            for suggest in suggests.split('\n'):
                if len(suggest) == 0: continue
                suggest_tokens = set([w for w in suggest.split() if len(w)>0])
                if qf_name_JPC in suggest_tokens:
                    suggest_tokens.remove(qf_name_JPC)
                if len(suggest_tokens)==1: retVal.update(suggest_tokens)
        return retVal
    
    @staticmethod
    def get_dict_suggest_word_vec(set_valid_suggest_words, w2v_model_path):
        model = Word2Vec.load(w2v_model_path)
        dict_wiki_w2v = {}
        for w, _ in model.wv.vocab.iteritems():
            if w.encode("utf-8") in set_valid_suggest_words:
                dict_wiki_w2v[w.encode("utf-8")] = model[w]
        return dict_wiki_w2v
    
    @staticmethod
    def _get_valid_sKeyWord_vec(doc, qf_name_JPC, suggest_word_vec): #helper for get_valid_sKeyWord_vec_col()
        for s_and_f in doc["suggest_f"]:
            suggest_tokens = set(re.split("\s+", s_and_f[0]))
            if qf_name_JPC in suggest_tokens:
                suggest_tokens.remove(qf_name_JPC)
            if (not len(suggest_tokens) == 1) or (not list(suggest_tokens)[0] in suggest_word_vec):
                continue
            return suggest_word_vec[list(suggest_tokens)[0]]
        return []
    
    @staticmethod
    def get_valid_sKeyWord_vec_col(sf, qf_name_JPC, suggest_word_vec): #suggest_f column must be created prior to calling
        return graphlab.SArray([DataUtility._get_valid_sKeyWord_vec(doc, qf_name_JPC, suggest_word_vec) for doc in sf])
    
    @staticmethod
    def _get_valid_sKeyWord(doc, qf_name_JPC, suggest_word_vec): #helper for get_valid_sKeyWord_vec_col()
        for s_and_f in doc["suggest_f"]:
            suggest_tokens = set(re.split("\s+", s_and_f[0]))
            if qf_name_JPC in suggest_tokens:
                suggest_tokens.remove(qf_name_JPC)
            if (not len(suggest_tokens) == 1) or (not list(suggest_tokens)[0] in suggest_word_vec):
                continue
            return list(suggest_tokens)[0]
        return ""
    
    @staticmethod
    def get_valid_sKeyWord_col(sf, qf_name_JPC, suggest_word_vec): #suggest_f column must be created prior to calling
        return graphlab.SArray([DataUtility._get_valid_sKeyWord(doc, qf_name_JPC, suggest_word_vec) for doc in sf])

    @staticmethod
    def get_doc_vec_col(d2v_model_path):
        model = Doc2Vec.load(d2v_model_path)
        return graphlab.SArray(list(model.docvecs))




