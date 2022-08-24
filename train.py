from torch.nn.functional import cross_entropy
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix, accuracy_score
import numpy as np

def get_scores(y_true, y_pred, labels=[0,1,2,3]):
    return accuracy_score(y_true, y_pred), \
            f1_score(y_true, y_pred, average='micro', labels=labels), \
            precision_score(y_true,y_pred, average='micro', labels=labels), \
            recall_score(y_true,y_pred, average='micro', labels=labels)

def eval(model, dataset, labels=[0,1,2,3], device='cpu'):
    y_preds = []
    y_trues = []
    for tokens, y_true in dataset:
        y_true.to(device)
        tokens.to(device)
        y_preds.append(model(tokens).argmax(axis=1).item())
        y_trues.append(y_true.item())

    conf_mat = confusion_matrix(y_trues, y_preds, labels=labels)
    print("Confusion Matrix:")
    print(conf_mat)

    return get_scores(y_trues, y_preds)
