from libtopic_nlp_iit.db_util import *
from libtopic_nlp_iit.data_util import *
from scipy.spatial.distance import cosine
from collections import Counter
import numpy as np
import pickle, re

class UnionFinder:
    
    class Node:
        def __init__(self):
            self.next = None
    
    def __init__(self):
        self.members = {}
    
    def _get_root(self, index):
        if not index in self.members:
            return None # not expected to happen
        cur = self.members[index]
        routine = []
        while cur.next is not None:
            routine.append(cur)
            cur = cur.next
        for n in routine:
            n.next = cur
        return cur
    
    def union_find(self, clusters):
        for cluster in clusters:
            new_members = set()
            roots = set()
            for d in cluster:
                if d not in self.members:
                    new_members.add(d)
                else:
                    roots.add(self._get_root(d))
            if len(roots) == 0:
                new_node = self.Node()
                for d in new_members:
                    self.members[d] = new_node
            elif len(roots) == 1:
                for d in new_members:
                    r = list(roots)[0]
                    self.members[d] = r
            else:
                new_node = self.Node()
                for r in roots:
                    r.next = new_node
                for d in new_members:
                    self.members[d] = new_node
    
        index_map = {}
        retVal = {}
        for d in self.members.keys():
            root = self._get_root(d)
            if root not in index_map:
                index_map[root] = len(index_map)
            retVal[d] = index_map[root]
        return retVal

class UnsupervisedPredictor:
    '''
        expected input doc attributes: ('web_id', 'url', 'content', 'entry', 'suggests', 'suggest_ranks', 'topic', 'topic_probability', 'token', 'vec_doc', 'suggest_f', 'label', 'sKeyWord_5', 'vec_sKeyWord_5', 'sKeyWord_8', 'vec_sKeyWord_8', 'sKeyWord_10', 'vec_sKeyWord_10', 'sKeyWord_wikiOnly_5', 'vec_sKeyWord_wikiOnly_5', 'sKeyWord_wikiOnly_8', 'vec_sKeyWord_wikiOnly_8', 'sKeyWord_wikiOnly_10', 'vec_sKeyWord_wikiOnly_10')
        This predictor expects db docs with at least attributes
        ('web_id', 'suggests', 'topic', 'token', 'vec_doc', 'vec_sKeyWord_5', 'sKeyWord_8', 'vec_sKeyWord_8', 'sKeyWord_10', 'vec_sKeyWord_10', 'sKeyWord_wikiOnly_5', 'vec_sKeyWord_wikiOnly_5', 'sKeyWord_wikiOnly_8', 'vec_sKeyWord_wikiOnly_8', 'sKeyWord_wikiOnly_10', 'vec_sKeyWord_wikiOnly_10', 'suggest_f').
        In addition, d2d_vsim and s2s_vsim dict are expected for fast similarity lookup
    '''
    def __init__(self, sim_mat, collection):
        self.sim_mat = sim_mat
        self.collection = collection
        self.qf_name = None
        self.qf_name_JPC = None
    
    def _get_topic_df(self, topic, attribs):
        attrib = ["web_id", "topic", "topic_probability",] + attribs
#        for k in collection.find_one().keys():
#            if k in attrib and self.sim_mat["sim_%s"%k] is None:
#                attrib.append(k)
        dict_attrib = {attr: 1 for attr in attrib}
        dict_attrib["_id"] = 0
        # if self.sim_mat["sim_token"] is None: dict_attrib["token"] = 1
        dbcursor = self.collection.find({"topic": topic}, dict_attrib).sort("topic_probability", pymongo.DESCENDING)
        return dbcursor2df(dbcursor)
    
    def predict_subtopics_on_vec(self, vec_col_name, lbds): #return a list of dict of same length of lbds
        retVal = [dict() for _ in range(len(lbds))]
        topics = self.collection.distinct(key="topic")
        for t in topics:
            df_t = self._get_topic_df(t, [vec_col_name])
            df_t[vec_col_name] = df_t[vec_col_name].apply(lambda v: pickle.loads(v))
            #docs with no valid s.k.w
            invalid_doc_ids = df_t["web_id"][df_t.apply(lambda doc: len(doc[vec_col_name])==0, axis=1)]
            if vec_col_name == "vec_doc": assert len(invalid_doc_ids)==0
            df_t = df_t[df_t.apply(lambda doc: len(doc[vec_col_name])>0, axis=1)]
            lbd_pairs = [list() for _ in range(len(lbds))] # pair lists for all thresholds
            for i in range(len(df_t)):
                for j in range(i+1, len(df_t)):
                    sim = 0
                    wid1, wid2 = df_t.iloc[i]["web_id"], df_t.iloc[j]["web_id"]
                    if not "sim_%s"%vec_col_name in self.sim_mat:
                        sim = 1 - cosine(df_t.iloc[i][vec_col_name], df_t.iloc[j][vec_col_name])
                    else:
                        sim = self.sim_mat["sim_%s"%vec_col_name][(wid1, wid2)]\
                            if wid1 < wid2 else self.sim_mat["sim_%s"%vec_col_name][(wid2, wid1)]
                    for k in range(len(lbds)):
                        if sim >= lbds[k]:
                            lbd_pairs[k].append((wid1, wid2))
            for i in range(len(lbds)):
                dict_webId2clusterId = UnionFinder().union_find(lbd_pairs[i])
                unpaired_id_index = max(list(dict_webId2clusterId.values()))+1 if len(dict_webId2clusterId)>0 else 0
                for wid in df_t["web_id"]:
                    if not wid in dict_webId2clusterId:
                        dict_webId2clusterId[wid] = unpaired_id_index
                        unpaired_id_index += 1
                for wid in invalid_doc_ids:
                    dict_webId2clusterId[wid] = -1
                retVal[i].update(dict_webId2clusterId)
            #some topics do not reach 30 documents in total; the next assertion is optionally controlled
            #assert len(to_append)>=30, "sKeyWord_vec to_append >= 30, " + str(len(to_append))
        return retVal
    
    def predict_subtopics_on_bow(self, vec_col_name, lbds): #return a list of dict of same length of lbds
        retVal = [dict() for _ in range(len(lbds))]
        topics = self.collection.distinct(key="topic")
        for t in topics:
            df_t = self._get_topic_df(t, [vec_col_name])
            df_t[vec_col_name] = df_t[vec_col_name].apply(lambda x: re.split("\s+", x.strip()))
            df_t[vec_col_name] = df_t[vec_col_name].apply(lambda x: Counter(x))
            lbd_pairs = [list() for _ in range(len(lbds))] # pair lists for all thresholds
            for i in range(len(df_t)):
                for j in range(i+1, len(df_t)):
                    sim = 0
                    wid1, wid2 = df_t.iloc[i]["web_id"], df_t.iloc[j]["web_id"]
                    if not "sim_%s"%vec_col_name in self.sim_mat:
                        sim = 1 - dict_cosine(df_t.iloc[i][vec_col_name], df_t.iloc[j][vec_col_name])
                    else:
                        sim = self.sim_mat["sim_%s"%vec_col_name][(wid1, wid2)]\
                            if wid1 < wid2 else self.sim_mat["sim_%s"%vec_col_name][(wid2, wid1)]
                for k in range(len(lbds)):
                    if sim >= lbds[k]:
                        lbd_pairs[k].append((wid1, wid2))
            for i in range(len(lbds)):
                dict_webId2clusterId = UnionFinder().union_find(lbd_pairs[i])
                unpaired_id_index = max(list(dict_webId2clusterId.values()))+1 if len(dict_webId2clusterId)>0 else 0
                for wid in df_t["web_id"]:
                    if not wid in dict_webId2clusterId:
                        dict_webId2clusterId[wid] = unpaired_id_index
                        unpaired_id_index += 1
                retVal[i].update(dict_webId2clusterId)
        #some topics do not reach 30 documents in total; the next assertion is optionally controlled
        #assert len(to_append)>=30, "sKeyWord_vec to_append >= 30, " + str(len(to_append))
        return retVal
    
    def predict_subtopics_on_topic_ranking(self, rlbds): #return a list of dict of same length of rlbds
        retVal = [dict() for _ in range(len(rlbds))]
        for k in range(len(rlbds)):
            pred = {}
            rlbd = rlbds[k]
            topics = self.collection.distinct(key="topic")
            for t in topics:
                df_t = self._get_topic_df(t, [])
                for i in range(rlbd):
                    pred[df_t["web_id"][i]] = -2
                for i in range(rlbd, len(df_t)):
                    pred[df_t["web_id"][i]] = -1
            retVal[k].update(pred)
        return retVal


