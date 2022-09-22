from torch.nn.functional import cross_entropy
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix, accuracy_score
import numpy as np
import math
import torch
import warnings
warnings.filterwarnings("ignore", message="Precision is ill-defined")

def get_scores(y_true, y_pred, num_classes):
    labels=[i for i in range(num_classes)]
    return  accuracy_score(y_true, y_pred), \
            precision_score(y_true,y_pred, average='macro', labels=labels), \
            recall_score(y_true,y_pred, average='macro', labels=labels), \
            f1_score(y_true, y_pred, average='macro', labels=labels)

def eval(model, dataset, indices, num_classes, device='cuda', is_verbose=False):
    '''
        return accuracy, precision, recall, f1-score
    '''
    # model.to(device)
    labels=[i for i in range(num_classes)]
    batch_size = 5
    iters = math.ceil(len(indices) // batch_size)
    y_preds = torch.empty((0,),device='cpu')
    y_trues = torch.empty((0,),device='cpu')
    for i in range(iters):
      tokens, y_true = dataset[indices[i*batch_size:(i+1)*batch_size]]
        
      tokens.to(device)
      y_trues.to('cpu')

      digits = model(tokens)
      y_preds = torch.hstack([y_preds,digits.argmax(dim=1).to('cpu')])
      y_trues = torch.hstack([y_trues,y_true])

    # tokens, y_trues = dataset
    # y_preds = model(tokens).argmax(dim=1)

    if is_verbose:
        conf_mat = confusion_matrix(y_trues, y_preds, labels=labels)
        print("Confusion Matrix:")
        print(conf_mat)

    return get_scores(y_trues, y_preds, num_classes)
