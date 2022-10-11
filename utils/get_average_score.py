import argparse

parser = argparse.ArgumentParser(description='plot training graph')
parser.add_argument('-f', '--file', help='training log file name')
args = parser.parse_args()
accuracy, precision, recall, f1_score = 0,0,0,0
with open(args.file, "r") as f:
    is_next = False
    for line in f.readlines():
        if is_next:
            temp = line.replace(","," ").split()
            this_accuracy, this_precision, this_recall, this_f1_score = float(temp[4]), float(temp[6]), float(temp[8]), float(temp[10])
            accuracy, precision, recall, f1_score = accuracy+this_accuracy, precision+this_precision, recall+this_recall, f1_score+this_f1_score
            is_next = False
        if "]]" in line:
            is_next = True
            

print("%.3f %.3f %.3f %.3f" % (accuracy/5, precision/5, recall/5, f1_score/5))