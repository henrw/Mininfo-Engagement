import string
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('log_file', type=string, help='training log file name')

args = parser.parse_args()
with open(args.log_file, 'r') as f:
    pass
    # for line
# loss_values = np.array([0.5, 0.3, 0.1])
# plt.plot(loss_values)
# plt.show()