from torch.nn.functional import cross_entropy
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix, accuracy_score
import numpy as np
import math
import torch
import warnings
warnings.filterwarnings("ignore", message="Precision is ill-defined")

def get_scores(y_true, y_pred, num_classes, is_verbose = False):
    labels=[i for i in range(num_classes)]
    if is_verbose:
        conf_mat = confusion_matrix(y_true, y_pred, labels=labels)
        print("Confusion Matrix:")
        print(conf_mat)
    return  accuracy_score(y_true, y_pred), \
            precision_score(y_true,y_pred, average='macro', labels=labels), \
            recall_score(y_true,y_pred, average='macro', labels=labels), \
            f1_score(y_true, y_pred, average='macro', labels=labels)

def eval(model, dataset, indices, num_classes, device, is_verbose=False):
    '''
        return accuracy, precision, recall, f1-score
    '''
    # model.to(device)
    batch_size = 5
    iters = math.ceil(len(indices) // batch_size)
    y_preds = torch.empty((0,),device=device)
    y_trues = torch.empty((0,),device=device)
    for i in range(iters):
      tokens, y_true = dataset[indices[i*batch_size:(i+1)*batch_size]]
      y_true =y_true.to(device)
    
      digits = model(tokens)
      y_preds = torch.hstack([y_preds,digits.argmax(dim=1)])
      y_trues = torch.hstack([y_trues,y_true])

    # tokens, y_trues = dataset
    # y_preds = model(tokens).argmax(dim=1)

    return get_scores(y_trues.to('cpu'), y_preds.to('cpu'), num_classes, is_verbose)

def eval_audio(model, dataset, indices, num_classes, device, is_verbose=False):
    '''
        return accuracy, precision, recall, f1-score
    '''
    # model.to(device)
    batch_size = 5
    iters = math.ceil(len(indices) // batch_size)
    y_preds = torch.empty((0,),device=device)
    y_trues = torch.empty((0,),device=device)
    for i in range(iters):
      _, waveforms, y_true = dataset[indices[i*batch_size:(i+1)*batch_size]]
      y_true = y_true.to(device)
      waveforms = waveforms.to(device)
    
      digits = model(waveforms)
      y_preds = torch.hstack([y_preds,digits.argmax(dim=1)])
      y_trues = torch.hstack([y_trues,y_true])

    # tokens, y_trues = dataset
    # y_preds = model(tokens).argmax(dim=1)

    return get_scores(y_trues.to('cpu'), y_preds.to('cpu'), num_classes, is_verbose)
