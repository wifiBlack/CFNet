import numpy as np


class ConfuseMatrixFor2Classes:
    def __init__(self) -> None:
        self.confuse_matrix = np.zeros((2, 2))
    def update(self, pred,target):
        self.confuse_matrix += calculate_confuse_matrix(pred,target)
    def get_scores(self):
        TP = self.confuse_matrix[0, 0]
        FP = self.confuse_matrix[1, 0]
        FN = self.confuse_matrix[0, 1]
        FP = self.confuse_matrix[1, 0]
        Recall = TP / (TP + FN + np.finfo(np.float32).eps)
        Precision = TP / (TP + FP + np.finfo(np.float32).eps)
        F1 = 2*Recall*Precision / (Recall + Precision + np.finfo(np.float32).eps)
        IoU = TP / (TP + FP + FN + np.finfo(np.float32).eps)
        score_dict = {"Recall": Recall, "Precision": Precision, "F1": F1, "IoU": IoU}
        return score_dict
def calculate_confuse_matrix(pred,target):
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    pred_flat = (pred_flat>0.5)
    target_flat = (target_flat>0.5)
    
    confuse_matrix = np.zeros((2, 2),dtype=int)
    TP = np.sum((pred_flat == 1) & (target_flat == 1))
    FP = np.sum((pred_flat == 1) & (target_flat == 0))
    FN = np.sum((pred_flat == 0) & (target_flat == 1))
    TN = np.sum((pred_flat == 0) & (target_flat == 0))
    
    confuse_matrix[0, 0] = TP
    confuse_matrix[0, 1] = FN
    confuse_matrix[1, 0] = FP
    confuse_matrix[1, 1] = TN
    return confuse_matrix