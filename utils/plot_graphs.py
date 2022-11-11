import string
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description='plot training graph')
parser.add_argument('-f', '--file', help='training log file name')
parser.add_argument('-t', '--title', nargs='?', help='plot title')
args = parser.parse_args()

fig, axs = plt.subplots(5,1, figsize = (10, 18))
with open(args.file, 'r') as f:
    loss = []
    val_accuracy = []
    train_accuracy = []
    # val_f1 = []
    # train_f1 = []
    for line in f.readlines():
        if "Fold " in line:
            fold = int(line.split()[1])
            if loss:
                epoch = [i for i in range(epoch+1)]
                axs[fold - 2].set_title('Fold '+str(fold-1))
                axs[fold - 2].plot(epoch, loss, color = "green")
                # axs[fold - 2].yaxis.set_ticks([0.5,1])

                ax2 = axs[fold - 2].twinx()
                ax2.plot(epoch, train_accuracy, color = 'blue')
                ax2.plot(epoch, val_accuracy, color = 'purple')
                ax2.yaxis.set_ticks([0,0.2,0.4,0.6,0.8,1])

                loss = []
                val_accuracy = []
                train_accuracy = []
        elif "(Epoch" in line:
            epoch = int(line.replace(')',' ').split()[1])
            loss.append(float(line.replace(')',' ').split()[-1]))
        elif "Training Set" in line:
            train_accuracy.append(float(line.replace(',',' ').split()[4]))
        elif "Validation Set" in line:
            val_accuracy.append(float(line.replace(',',' ').split()[4]))
    
    epoch = [i for i in range(epoch+1)]
    axs[fold - 1].set_title('Fold '+str(fold))
    axs[fold - 1].plot(epoch, loss, label="loss", color = "green")

    ax2 = axs[fold - 1].twinx()
    ax2.plot(epoch, train_accuracy, label="train_accuracy", color = 'blue')
    ax2.plot(epoch, val_accuracy, label="val_accuracy", color = 'purple')
    axs[fold - 1].set_xlabel("epochs")
    ax2.yaxis.set_ticks([0,0.2,0.4,0.6,0.8,1])

lines_labels = [axs[-1].get_legend_handles_labels()]+[ax2.get_legend_handles_labels()]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend(lines, labels)
if args.title:
    fig.suptitle(args.title, fontsize=16)

plt.savefig("visualization/"+args.file.replace(".log",".png").replace("train_logs/",""))