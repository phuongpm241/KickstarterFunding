import matplotlib.pyplot as plt
import pylab as pl
import numpy as np 
from plotData import * 

epochs = [1, 5, 10, 20, 30, 50]
train_acc = [0.8216, 0.8628, 0.8711, 0.8721, 0.8729, 0.8733]
val_acc = [0.8290, 0.8749, 0.8752, 0.8760, 0.8765, 0.8780]
xlabel = 'Epochs'
ylabel = 'Accuracy'
title = 'Neural Network with Varying Training Epochs'
plotLine(epochs, train_acc, val_acc, xlabel, ylabel, title, None)

optimizer = ['Adam', 'Adamax', 'Adagrad', 'SGD', 'Adadelta']
train_acc = [0.8692, 0.8710, 0.8721, 0.8397, 0.8666]
val_acc = [0.8766, 0.8766, 0.8760, 0.8523, 0.8748]
xlabel = 'Optimizer'
ylabel = 'Accuracy'
title = 'Neural Network with Varying Optimizer'
plotModels(optimizer, train_acc, val_acc, xlabel, ylabel, title)

dropout_prob = [0.0, 0.05, 0.1, 0.2, 0.5]
train_acc = [0.8720, 0.8714, 0.8710, 0.8686, 0.8594]
val_acc = [0.8780, 0.8771, 0.8766, 0.8758, 0.8758]
xlabel = 'Dropout Probability'
ylabel = 'Accuracy'
title = 'Neural Network with Varying Dropout Probability'
plotLine(dropout_prob, train_acc, val_acc, xlabel, ylabel, title, None)

units = [10, 50, 100, 200, 500, 1000, 5000, 10000]
train_acc = [0.8324, 0.8709, 0.8715, 0.8715, 0.8704, 0.8710, 0.8699, 0.8688]
val_acc = [0.8745, 0.8771, 0.8770, 0.8777, 0.8761, 0.8755, 0.8766, 0.8759]
xlabel = 'Hidden Units'
ylabel = 'Accuracy'
title = 'Neural Network with Varying Hidden Units'
plotLine(units, train_acc, val_acc, xlabel, ylabel, title, None)

layers = [2, 3, 4, 5, 6]
train_acc = [0.8725, 0.8707, 0.8705, 0.8702, 0.8699]
val_acc = [0.8763, 0.8770, 0.8762, 0.8764, 0.8758]
xlabel = 'Hidden Layers'
ylabel = 'Accuracy'
title = 'Neural Network with Varying Hidden Layers'
plotLine(layers, train_acc, val_acc, xlabel, ylabel, title, None)



