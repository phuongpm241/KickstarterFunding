import matplotlib.pyplot as plt
import pylab as pl
import numpy as np 

def plotModels(x, train_acc, val_acc, xlabel, ylabel, title):
	ind = np.arange(len(x))
	width = 0.2
	plt.bar(ind, train_acc, width, label='Training Accuracy')
	plt.bar(ind + width, val_acc, width, label='Validation Accuracy')
	plt.ylabel(ylabel)
	plt.xlabel(xlabel)
	plt.title(title)
	plt.xticks(ind + width / 2, x)
	plt.legend(loc='best')
	plt.show() 

def plotParameters(x, y, title):
	plt.plot(x,y,'-')
	plt.xlabel('x')
	plt.ylabel('y')
	plt.title(title)
	plt.show()

##############################
##							##
##		 PEGASOS SVM		##
##							##
##############################

lambda_val = ['2e-10', '2e-8', '2e-6', '2e-4', '2e-2', '2e-1', '2e0', '2e1', '2e2', '2e4']
train_acc = []
val_acc = []

##############################
##							##
##		  KERNEL SVM		##
##							##
##############################


##############################
##							##
##	  LOGISTIC REGRESSION	##
##							##
##############################

regularizer = ['liblinear', 'lbgfs', 'saga', 'sag']
train_acc = [0.791, 0.792, 0.745, 0.751]
val_acc = [0.788, 0.789, 0.744, 0.750]

C = [0.01, 0.1, 1, 10, 100]
train_acc = [0.792, 0.791, 0.792, 0.791, 0.792]
val_acc = [0.788, 0.788, 0.789, 0.788, 0.789] 

iterations = [100, 500, 1000]
train_acc = [0.792, 0.792, 0.792]
val_acc = [0.789, 0.789, 0.789]

##############################
##							##
##   K LOGISTIC REGRESSION  ##
##							##
##############################


##############################
##							##
##	    NEURAL NETWORKS		##
##							##
##############################

# 3 Hidden Layer, Adamax, ReLU, Random Init, 0.1 Dropout, 50 Epochs
# Varying Neurons 

# Input -> 100 -> 50 -> 10 -> Output
# Input -> 1000 -> 50 -> 10 -> Output 
# Input -> 5000 -> 50 -> 10 -> Output 

units = [100, 1000, 5000]
train_acc = [0.875, 0.875, 0.875]
val_acc = [0.876, 0.871, 0.872]

# Input -> 100 -> 50 -> 10 -> Output
# Input -> 1000 -> 500 -> 100 -> Output 
# Input -> 5000 -> 2500 -> 500 -> Output 

units = [100, 1000, 5000]
train_acc = [0.875, 0.875, 0.686]
val_acc = [0.876, 0.872, 0.682]

# 3 Hidden Layer, Adamax, ReLU, Random Init, 100 Neurons, 50 Epochs
# Dropout 
dropout = [0, 0.05, 0.1, 0.2, 0.3, 0.5]
train_acc = [0.876, 0.876, 0.875, 0.874, 0.872, 0.861]
val_acc = [0.876, 0.875, 0.876, 0.876, 0.877, 0.868]

# 3 Hidden Layer, ReLU, Random Init, 100 Neurons, 50 Epochs, 0.05 Dropout
# Optimizer 
optimizer = ['Adam', 'Adamax', 'Adagrad', 'SGD', 'Adadelta']
train_acc = [0.875, 0.876, 0.876, 0.858, 0.875]
val_acc = [0.876, 0.876, 0.877, 0.874, 0.874]

# Adamax, ReLU, Random Init, 100 Neurons, 0.05 Dropout, 50 Epochs
# Hidden Layer 

# Input -> 100 -> 50 -> 10 -> Output 
# Input -> 200 -> 100 -> 50 -> 25 -> 10 -> Output 
# Input -> 500 -> 250 -> 150 -> 100 -> 50 -> 25 -> 10 -> Output

hidden_layer = [3, 5, 7]
train_acc = [0.876, 0.876, 0.875]
val_acc = [0.875, 0.878, 0.878]

# 3 Hidden Layer, Adagrad, ReLU, Random Init, 100 Neurons, 50 Epochs, 0.05 Dropout
# Batch Normalization 
normalization = [0, 1]
train_acc = [0.876, 0.870]
val_acc = [0.875, 0.576]

# 3 Hidden Layer, Adagrad Random Init, 100 Neurons, 50 Epochs, 0.05 Dropout
activation_fn = ['relu', 'tanh', 'sigmoid']
train_acc = [0.876, 0.871, 0.873]
val_acc = [0.875, 0.863, 0.876]



