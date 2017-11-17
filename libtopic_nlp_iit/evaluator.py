from collections import Counter
import numpy as np

def label2binary(list_label, df=3):
    label_counts = Counter(list_label)
    retVal = [(x==-2) or ((not x=="NULL") and (not x==-1) and label_counts[x]>=df) for x in list_label]
    return retVal

class Evaluator:
    def __init__(self, lists_subtopic, lists_label): #list-like inputs
        assert len(lists_subtopic) == len(lists_label), "evaluator input size mismatch!"
        self.lists_prediction = np.array([label2binary(l) for l in lists_subtopic])
        self.lists_label = np.array([label2binary(l) for l in lists_label])
        self.con_tensor = self._get_confusion_tensor()
    
    def _get_confusion_mat(self, list_prediction, list_label):
        assert len(list_prediction) == len(list_label), "evaluator input size mismatch!"
        retVal = {"TP": None, "FP": None, "TN": None, "FN": None}
        retVal["TP"] = np.count_nonzero((list_prediction==1)&(list_label==1))
        retVal["FP"] = np.count_nonzero((list_prediction==1)&(list_label==0))
        retVal["TN"] = np.count_nonzero((list_prediction==0)&(list_label==0))
        retVal["FN"] = np.count_nonzero((list_prediction==0)&(list_label==1))
        assert retVal["TP"] + retVal["FP"] + retVal["TN"] + retVal["FN"] == len(list_label), "invalid confusion matrix"
        return retVal

    def _get_confusion_tensor(self):
        retVal = {"TP":[], "FP":[], "TN":[], "FN":[]}
        for prediction, label in zip(self.lists_prediction, self.lists_label):
            mat = self._get_confusion_mat(prediction, label)
            retVal["TP"].append(mat["TP"])
            retVal["FP"].append(mat["FP"])
            retVal["TN"].append(mat["TN"])
            retVal["FN"].append(mat["FN"])
        return retVal
    
    def get_accuracy(self):
        total = 0
        accurate = 0
        for prediction, label in zip(self.lists_prediction, self.lists_label):
            accurate += np.count_nonzero(prediction==label)
            total += len(label)
        return float(accurate) / total
    
    def get_macro_pre_rec(self):
        TPs = np.array(self.con_tensor["TP"])
        FPs = np.array(self.con_tensor["FP"])
        TNs = np.array(self.con_tensor["TN"])
        FNs = np.array(self.con_tensor["FN"])
        mask_TP_plus_FP = (TPs+FPs)!=0
        precisions = TPs.astype(np.float32)[mask_TP_plus_FP] / (TPs+FPs)[mask_TP_plus_FP]
        mask_TP_plus_FN = (TPs+FNs)!=0
        recalls = TPs.astype(np.float32)[mask_TP_plus_FN] / (TPs+FNs)[mask_TP_plus_FN]
        return precisions.mean() if len(precisions)>0 else 0, recalls.mean() if len(recalls)>0 else 0
    
    def get_micro_pre_rec(self):
        TPs = np.array(self.con_tensor["TP"])
        FPs = np.array(self.con_tensor["FP"])
        TNs = np.array(self.con_tensor["TN"])
        FNs = np.array(self.con_tensor["FN"])
        precision = TPs.sum().astype(np.float32) / (TPs.sum() + FPs.sum()) if TPs.sum() + FPs.sum() > 0 else 0
        recall = TPs.sum().astype(np.float32) / (TPs.sum() + FNs.sum()) if TPs.sum() + FNs.sum() > 0 else 0
        return precision, recall




