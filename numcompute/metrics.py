import numpy as np


class Classification:
    @staticmethod
    def confusion_matrix(y_true, y_pred):
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        return tp, tn, fp, fn
    
    @staticmethod
    def accuracy(y_true, y_pred):
        tp, tn, fp, fn = Classification.confusion_matrix(y_true, y_pred)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        return accuracy
    
    @staticmethod
    def precision(y_true, y_pred):
        tp, tn, fp, fn = Classification.confusion_matrix(y_true, y_pred)
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        return precision
    
    @staticmethod
    def recall(y_true, y_pred):
        tp, tn, fp, fn = Classification.confusion_matrix(y_true, y_pred)
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        return recall
    
    @staticmethod
    def f1(y_true, y_pred):
        precision = Classification.precision(y_true, y_pred)
        recall = Classification.recall(y_true, y_pred)
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
        return f1


class Regression:
    @staticmethod
    def mse(y_true, y_pred):
        return np.mean(np.square(y_true - y_pred))
