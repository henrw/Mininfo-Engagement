device = "cuda:2"
from dataset import YouTubeDataset
from models.single_modality_classifiers import Wav2VecClassifier

import torch
import math
import os
import time
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from inference import eval_audio, get_scores
from torch.nn.functional import cross_entropy

def train_model(model, dataset, learning_rate, lr_decay, weight_decay, batch_size, num_epochs, device, isCheckpoint=False, train_val_split = None, isVerbose=True):
    loss_history = []

    model.to(device)
    dataset.audio = dataset.audio.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), learning_rate, weight_decay=weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: lr_decay ** epoch
    )

    # sample minibatch data
    if not train_val_split:
      train_ids = [i for i in range(len(dataset))]
      val_ids = None
    else:
      train_ids, val_ids = train_val_split

    iter_per_epoch = math.ceil(len(train_ids) // batch_size)
    class_weights = torch.tensor(compute_class_weight(class_weight='balanced', classes=np.arange(model.num_classes), y=dataset.label[train_ids].numpy()), dtype=torch.float, device=device)
    loss_fn = torch.nn.NLLLoss(weight = class_weights)
    # loss_fn = cross_entropy
    
    for i in range(num_epochs):
        start_t = time.time()
        local_hist = []
        correct_cnt = 0
        y_preds = torch.empty((0,),device=device)
        y_trues = torch.empty((0,),device=device)
        for j in range(iter_per_epoch):
            _, waveforms, y_true = dataset[train_ids[j * batch_size: (j + 1) * batch_size]]

            waveforms = waveforms.to(device)
            y_true = y_true.to(device)

            optimizer.zero_grad()

            digits = model(waveforms)
            y_preds = torch.hstack([y_preds,digits.argmax(dim=1)])
            y_trues = torch.hstack([y_trues,y_true])

            probs = torch.nn.LogSoftmax(dim=1)(digits)
            loss = loss_fn(probs,y_true)
            loss.backward()

            local_hist.append(loss.item())
            optimizer.step()

        end_t = time.time()

        loss_mean = np.array(local_hist).mean()
        loss_history.append(loss_mean)
            
        print(
            f"(Epoch {i}), time: {end_t - start_t:.1f}s, loss: {loss_mean:.3f}"
        )
        if isVerbose:
            train_accuracy, train_precision, train_recall, train_f1 = get_scores(y_trues.to('cpu'), y_preds.to('cpu'), model.num_classes) # This is an aggregated result due to GPU size limit
            print(f"    Training Set - accuracy: {train_accuracy:.2f}, precision: {train_precision:.2f}, recall: {train_recall:.2f}, f1-score: {train_f1:.2f},")
            if val_ids is not None:
                val_accuracy, val_precision, val_recall, val_f1 = eval_audio(model, dataset, val_ids, num_classes, device, is_verbose = (loss_mean < 0.5))
                print(f"    Validation Set - accuracy: {val_accuracy:.2f}, precision: {val_precision:.2f}, recall: {val_recall:.2f}, f1-score: {val_f1:.2f},")
        if i%200 == 0 and isCheckpoint:
          dir = "checkpoints"
          if not os.path.exists(dir):
            os.mkdir(dir)
          file = f"epoch{i}.pt"
          path = dir+'/'+file
          torch.save({
                      'epoch': i,
                      'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(),
                      'loss': loss_mean,
                      }, path)

        lr_scheduler.step()

        if loss_mean < 0.5:
          break
    
    return loss_history

from sklearn.model_selection import KFold
import torch

def train_model_cv5(model, dataset):
    loss_hist = []
    kf = KFold(n_splits=5)
    cnt = 1
    for train_index, val_index in kf.split(dataset):
        model.reset()
        model.base.requires_grad = False
        print("Fold "+str(cnt)+" (val", val_index[0],"-",str(val_index[-1])+")")
        loss_hist_fold = train_model(model, device = device, dataset=dataset, train_val_split=(train_index, val_index),learning_rate=1e-5, lr_decay=0.99, weight_decay=1e-4, batch_size=10, num_epochs=300, isCheckpoint = False, isVerbose = True)
        loss_hist.append(loss_hist_fold)
        cnt += 1
    return loss_hist


splits = [10,20,30]
num_classes = len(splits)+1

dataset = YouTubeDataset(splits)
model = Wav2VecClassifier(num_classes)
train_model_cv5(model, dataset)

splits = [10,20]
num_classes = len(splits)+1

dataset = YouTubeDataset(splits)
model = Wav2VecClassifier(num_classes)
train_model_cv5(model, dataset)

splits = [10]
num_classes = len(splits)+1

dataset = YouTubeDataset(splits)
model = Wav2VecClassifier(num_classes)
train_model_cv5(model, dataset)


