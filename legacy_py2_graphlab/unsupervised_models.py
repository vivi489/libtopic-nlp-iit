# -*- coding: utf-8 -*-
import graphlab, os, pickle, sys, re
from collections import defaultdict, Counter
from scipy.spatial.distance import cosine
from operator import itemgetter

from data_util import *
import numpy as np

qf_name_dict = dict([("kafunsyo", "花粉症"), ("kekkon", "結婚"), ("syukatsu", "就活"), ("3dprinter", "3dプリンタ"), ("mansion", "マンション")])

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
        input dataset: ('web_id', 'url', 'content', 'entry', 'suggests', 'suggest_ranks', 'topic', 'topic_probability', 'token', 'vec_doc', 'suggest_f', 'label', 'sKeyWord_5', 'vec_sKeyWord_5', 'sKeyWord_8', 'vec_sKeyWord_8', 'sKeyWord_10', 'vec_sKeyWord_10')
        This predictor expects sframes with at least columns of
        ('web_id', 'suggests', 'topic', 'token', 'vec_doc', 'vec_sKeyWord_5', 'vec_sKeyWord_8', 'vec_sKeyWord_10', 'suggest_f').
        In addition, d2d_vsim and s2s_vsim dict are expected for fast similarity lookup
    '''
    def __init__(self, d2d_vsim=None, s2s_vsim=None, d2d_wsim=None, list_sframe_topic=None):
        self.d2d_vsim = d2d_vsim
        self.d2d_wsim = d2d_wsim
        self.s2s_vsim = s2s_vsim
        self.list_sframe_topic = list_sframe_topic
        self.qf_name = None
        self.qf_name_JPC = None
    
    def predict_subtopics_on_vec_doc(self, SIM_LB=0.50): #return a list of SArrays
        retVal = []
        for sf_t in self.list_sframe_topic:
            pairs = []
            for i in xrange(len(sf_t)):
                for j in xrange(i+1, len(sf_t)):
                    cur_sim = 0
                    if self.d2d_vsim is None:
                        cur_sim = 1 - cosine(sf_t[i]["vec_doc"], sf_t[j]["vec_doc"])
                    else:
                        cur_sim = self.d2d_vsim[(sf_t[i]["web_id"], sf_t[j]["web_id"])] if sf_t[i]["web_id"]<sf_t[j]["web_id"]\
                            else self.d2d_vsim[(sf_t[j]["web_id"], sf_t[i]["web_id"])]
                    if cur_sim >= SIM_LB:
                        pairs.append((sf_t[i]["web_id"], sf_t[j]["web_id"]))
            else:
                pass
            dict_webId2clusterId = UnionFinder().union_find(pairs)
            unpaired_id_index = np.array(dict_webId2clusterId.values()).max()+1 if len(dict_webId2clusterId)>0 else 0
            for i in xrange(len(sf_t)):
                if not sf_t[i]["web_id"] in dict_webId2clusterId:
                    dict_webId2clusterId[sf_t[i]["web_id"]] = unpaired_id_index
                    unpaired_id_index += 1
            #retVal.append(sf_t.apply(lambda d: dict_webId2clusterId[d["web_id"]]))
            to_append = graphlab.SArray([dict_webId2clusterId[d["web_id"]] for d in sf_t])
            #assert len(to_append)>=30, "vec_doc to_append >= 30, " + str(len(to_append))
            retVal.append(to_append)
        return retVal
    
    def predict_subtopics_on_sKeyWord_vec(self, vec_sKeyWord_col_name, SIM_LB=0.80): #return a list of SArrays
        retVal = []
        for sf_t in self.list_sframe_topic:
            sf_all = sf_t
            invalid_doc_ids = sf_all[sf_all.apply(lambda doc: len(doc[vec_sKeyWord_col_name])==0)]["web_id"]
            sf_t = sf_t[sf_t.apply(lambda doc: len(doc[vec_sKeyWord_col_name])>0)]
            pairs = []
            for i in xrange(len(sf_t)):
                for j in xrange(i+1, len(sf_t)):
                    cur_sim = 0
                    if self.s2s_vsim is None:
                        cur_sim = 1 - cosine(sf_t[i][vec_sKeyWord_col_name], sf_t[j][vec_sKeyWord_col_name])
                    else:
                        cur_sim = self.s2s_vsim[(sf_t[i]["web_id"], sf_t[j]["web_id"])] if sf_t[i]["web_id"]<sf_t[j]["web_id"]\
                            else self.s2s_vsim[(sf_t[j]["web_id"], sf_t[i]["web_id"])]
                    if cur_sim >= SIM_LB:
                        pairs.append((sf_t[i]["web_id"], sf_t[j]["web_id"]))
            dict_webId2clusterId = UnionFinder().union_find(pairs)
            for invalid_doc_id in invalid_doc_ids:
                dict_webId2clusterId[invalid_doc_id] = -1
            unpaired_id_index = np.array(dict_webId2clusterId.values()).max()+1 if len(dict_webId2clusterId)>0 else 0
            for doc in sf_all:
                if not doc["web_id"] in dict_webId2clusterId:
                    dict_webId2clusterId[doc["web_id"]] = unpaired_id_index
                    unpaired_id_index += 1
            #retVal.append(sf_t.apply(lambda d: dict_webId2clusterId[d["web_id"]]))
            to_append = graphlab.SArray([dict_webId2clusterId[d["web_id"]] for d in sf_all])
            #some topics do not reach 30 documents in total; the next assertion is optionally controlled
            #assert len(to_append)>=30, "sKeyWord_vec to_append >= 30, " + str(len(to_append))
            retVal.append(to_append)
        return retVal

    def predict_subtopics_on_bow(self, SIM_LB=0.50): #return a list of SArrays
        retVal = []
        for sf_t in self.list_sframe_topic:
            pairs = []
            if self.d2d_wsim is None:
                sf_t["word_count"] = graphlab.text_analytics.count_words(sf_t["token"])
            for i in xrange(len(sf_t)):
                for j in xrange(i+1, len(sf_t)):
                    cur_sim = 0
                    if self.d2d_wsim is None:
                        cur_sim = 1 - graphlab.toolkits.distances.cosine(sf_t[i]["word_count"], sf_t[j]["word_count"])
                    else:
                        cur_sim = self.d2d_wsim[(sf_t[i]["web_id"], sf_t[j]["web_id"])] if sf_t[i]["web_id"]<sf_t[j]["web_id"]\
                            else self.d2d_wsim[(sf_t[j]["web_id"], sf_t[i]["web_id"])]
                    if cur_sim >= SIM_LB:
                        pairs.append((sf_t[i]["web_id"], sf_t[j]["web_id"]))
            else:
                pass
            dict_webId2clusterId = UnionFinder().union_find(pairs)
            unpaired_id_index = np.array(dict_webId2clusterId.values()).max()+1 if len(dict_webId2clusterId)>0 else 0
            for i in xrange(len(sf_t)):
                if not sf_t[i]["web_id"] in dict_webId2clusterId:
                    dict_webId2clusterId[sf_t[i]["web_id"]] = unpaired_id_index
                    unpaired_id_index += 1
            #retVal.append(sf_t.apply(lambda d: dict_webId2clusterId[d["web_id"]]))
            to_append = graphlab.SArray([dict_webId2clusterId[d["web_id"]] for d in sf_t])
            #assert len(to_append)>=30, "bow to_append >= 30, " + str(len(to_append))
            retVal.append(to_append)
        return retVal

    
    def predict_subtopics_on_topic_ranking(self, RANK_LB=0): #return a list of SArrays
        retVal = []
        for sf_t in self.list_sframe_topic:
            retVal.append(graphlab.SArray([1]*RANK_LB + range(2, 2+len(sf_t)-RANK_LB)))
        return retVal

    @staticmethod
    def get_binary_classes_from_labels(sa, df=3): #takes an SArray of either manual labels or subtopic predictions
        #return an SArray
        label_counter = Counter(sa)
        retVal = graphlab.SArray([(not x=="NULL") and (not x==-1) and label_counter[x]>=df for x in sa])
        return retVal

class Evaluator:
    def __init__(self, sa_list_prediction=None, sa_list_label=None):
        self.sa_list_prediction = [UnsupervisedPredictor.get_binary_classes_from_labels(sa) for sa in sa_list_prediction]
        self.sa_list_label = [UnsupervisedPredictor.get_binary_classes_from_labels(sa) for sa in sa_list_label]
        assert len(sa_list_prediction) == len(sa_list_label), "evaluator input size mismatch!"
        self.combo_con_mat = self._get_confusion_mat_combo()
    
    def _get_confusion_mat(self, sa_prediction, sa_label):
        retVal = {"TP": None, "FP": None, "TN": None, "FN": None}
        retVal["TP"] = ((sa_prediction==1)&(sa_label==1)).nnz()
        retVal["FP"] = ((sa_prediction==1)&(sa_label==0)).nnz()
        retVal["TN"] = ((sa_prediction==0)&(sa_label==0)).nnz()
        retVal["FN"] = ((sa_prediction==0)&(sa_label==1)).nnz()
        assert retVal["TP"] + retVal["FP"] + retVal["TN"] + retVal["FN"] == len(sa_label), "invalid confusion matrix"
        return retVal
    
    def _get_confusion_mat_combo(self):
        retVal = {"TP":[], "FP":[], "TN":[], "FN":[]}
        for sa_prediction, sa_label in zip(self.sa_list_prediction, self.sa_list_label):
            assert len(sa_prediction) == len(sa_label), str(len(sa_prediction)) + "," + str(len(sa_label))
            mat = self._get_confusion_mat(sa_prediction, sa_label)
            retVal["TP"].append(mat["TP"])
            retVal["FP"].append(mat["FP"])
            retVal["TN"].append(mat["TN"])
            retVal["FN"].append(mat["FN"])
        return retVal
    
    def get_accuracy(self):
        total = 0
        accurate = 0
        for sa_prediction, sa_label in zip(self.sa_list_prediction, self.sa_list_label):
            accurate += (sa_prediction==sa_label).nnz()
            total += len(sa_label)
        return float(accurate) / total
    
    def get_macro_pre_rec(self):
        TPs = np.array(self.combo_con_mat["TP"])
        FPs = np.array(self.combo_con_mat["FP"])
        TNs = np.array(self.combo_con_mat["TN"])
        FNs = np.array(self.combo_con_mat["FN"])
        mask_TP_FP = (TPs+FPs)!=0
        precisions = TPs.astype(np.float32)[mask_TP_FP] / (TPs+FPs)[mask_TP_FP]
        mask_TP_FN = (TPs+FNs)!=0
        recalls = TPs.astype(np.float32)[mask_TP_FN] / (TPs+FNs)[mask_TP_FN]
        return precisions.mean() if len(precisions)>0 else 0, recalls.mean() if len(recalls)>0 else 0
    
    def get_micro_pre_rec(self):
        TPs = np.array(self.combo_con_mat["TP"])
        FPs = np.array(self.combo_con_mat["FP"])
        TNs = np.array(self.combo_con_mat["TN"])
        FNs = np.array(self.combo_con_mat["FN"])
        precision = TPs.sum().astype(np.float32) / (TPs.sum() + FPs.sum()) if TPs.sum() + FPs.sum() > 0 else 0
        recall = TPs.sum().astype(np.float32) / (TPs.sum() + FNs.sum()) if TPs.sum() + FNs.sum() > 0 else 0
        return precision, recall





