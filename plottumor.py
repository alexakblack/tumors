import argparse
import numpy as np

ap = argparse.ArgumentParser()

ap.add_argument(
    "-p",
    "--plot-history-load-path",
    required=True,
    help="path (.csv) to load training history from",
)

args = vars(ap.parse_args())


import csv
import matplotlib.pyplot as plt
import matplotlib.axes as axes

accuracy = []
val_accuracy = []
loss = []
val_loss = []

try:
	with open(args["plot_history_load_path"], "r") as f:
		history = list(csv.reader(f, delimiter=","))
		# for i in range(1, len(history)-1):
		# 	for j in range(0,4):
		# 		history[i][j] = float(history[i][j])
		for i in range(1, len(history)-1):
			accuracy.append(float(history[i][1]))
			val_accuracy.append(float(history[i][3]))
			loss.append(float(history[i][2]))
			val_loss.append(float(history[i][4]))
except FileNotFoundError:
	ap.error("invalid path to training history file")


train_accuracy_axis = accuracy
validation_accuracy_axis = val_accuracy
train_loss_axis = loss
validation_loss_axis = val_loss

epoch_axis = len(train_accuracy_axis)

ta, = plt.plot(np.arange(0,len(train_accuracy_axis)), train_accuracy_axis, 'b', label='training accuracy')
va, = plt.plot(np.arange(0, len(validation_accuracy_axis)), validation_accuracy_axis, 'r', label='validation accuracy')
plt.title('Accuracy')
plt.legend((ta, va), ("Training Accuracy", "Validation Accuracy"))
plt.show()

ta, = plt.plot(np.arange(0, len(train_loss_axis)), train_loss_axis, 'b', label='training loss')
va, = plt.plot(np.arange(0, len(validation_loss_axis)), validation_loss_axis, 'r', label='validation loss')
plt.title('Loss')
plt.yticks(np.arange(0, 3))
plt.legend((ta, va), ("Training Loss", "Validation Loss"))
plt.show()
