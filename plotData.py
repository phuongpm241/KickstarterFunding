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
train_acc = [0.652, 0.652, 0.652, 0.652, 0.668, 0.748, 0.760, 0.763, 0.801, 0.559]
val_acc = [0.652, 0.652, 0.652, 0.652, 0.666, 0.747, 0.760, 0.763, 0.796, 0.557]
xlabel = 'Regularization Parameter'
ylabel = 'Accuracy'
title = 'SVM with Varying Regularization Parameter'
plotModels(lambda_val, train_acc, val_acc, xlabel, ylabel, title)

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
xlabel = 'Solver'
ylabel = 'Accuracy'
title = 'Logistic Regression with Varying Solver'
plotModels(regularizer, train_acc, val_acc, xlabel, ylabel, title)

C = [0.01, 0.1, 1, 10, 100]
train_acc = [0.792, 0.791, 0.792, 0.791, 0.792]
val_acc = [0.788, 0.788, 0.789, 0.788, 0.789] 
xlabel = 'Regularization Parameter'
ylabel = 'Accuracy'
title = 'Logistic Regression with Varying Regularization Parameter'
plotModels(C, train_acc, val_acc, xlabel, ylabel, title)

iterations = [100, 500, 1000]
train_acc = [0.792, 0.792, 0.792]
val_acc = [0.789, 0.789, 0.789]
xlabel = 'Iterations'
ylabel = 'Accuracy'
title = 'Logistic Regression with Varying Iterations'
plotModels(iterations, train_acc, val_acc, xlabel, ylabel, title)

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
xlabel = 'Hidden Units'
ylabel = 'Accuracy'
title = 'Neural Network with Varying Hidden Units'
plotModels(units, train_acc, val_acc, xlabel, ylabel, title)

# Input -> 100 -> 50 -> 10 -> Output
# Input -> 1000 -> 500 -> 100 -> Output 
# Input -> 5000 -> 2500 -> 500 -> Output 

units = [100, 1000, 5000]
train_acc = [0.875, 0.875, 0.686]
val_acc = [0.876, 0.872, 0.682]
xlabel = 'Hidden Units'
ylabel = 'Accuracy'
title = 'Neural Network with Varying Hidden Units'
plotModels(units, train_acc, val_acc, xlabel, ylabel, title)

# 3 Hidden Layer, Adamax, ReLU, Random Init, 100 Neurons, 50 Epochs
# Dropout 
dropout = [0, 0.05, 0.1, 0.2, 0.3, 0.5]
train_acc = [0.876, 0.876, 0.875, 0.874, 0.872, 0.861]
val_acc = [0.876, 0.875, 0.876, 0.876, 0.877, 0.868]
xlabel = 'Dropout Probability'
ylabel = 'Accuracy'
title = 'Neural Network with Varying Dropout Probability'
plotModels(dropout, train_acc, val_acc, xlabel, ylabel, title)

# 3 Hidden Layer, ReLU, Random Init, 100 Neurons, 50 Epochs, 0.05 Dropout
# Optimizer 
optimizer = ['Adam', 'Adamax', 'Adagrad', 'SGD', 'Adadelta']
train_acc = [0.875, 0.876, 0.876, 0.858, 0.875]
val_acc = [0.876, 0.876, 0.877, 0.874, 0.874]
xlabel = 'Optimizer'
ylabel = 'Accuracy'
title = 'Neural Network with Varying Optimizer'
plotModels(optimizer, train_acc, val_acc, xlabel, ylabel, title)

# Adamax, ReLU, Random Init, 100 Neurons, 0.05 Dropout, 50 Epochs
# Hidden Layer 

# Input -> 100 -> 50 -> 10 -> Output 
# Input -> 200 -> 100 -> 50 -> 25 -> 10 -> Output 
# Input -> 500 -> 250 -> 150 -> 100 -> 50 -> 25 -> 10 -> Output

hidden_layer = [3, 5, 7]
train_acc = [0.876, 0.876, 0.875]
val_acc = [0.875, 0.878, 0.878]
xlabel = 'Hidden Layers'
ylabel = 'Accuracy'
title = 'Neural Network with Varying Hidden Layers'
plotModels(hidden_layer, train_acc, val_acc, xlabel, ylabel, title)

# 3 Hidden Layer, Adagrad, ReLU, Random Init, 100 Neurons, 50 Epochs, 0.05 Dropout
# Batch Normalization 
normalization = ['false', 'true']
train_acc = [0.876, 0.870]
val_acc = [0.875, 0.576]
xlabel = 'Batch Normalization'
ylabel = 'Accuracy'
title = 'Neural Network with/without Batch Normalization'
plotModels(normalization, train_acc, val_acc, xlabel, ylabel, title)

# 3 Hidden Layer, Adagrad Random Init, 100 Neurons, 50 Epochs, 0.05 Dropout
activation_fn = ['relu', 'tanh', 'sigmoid']
train_acc = [0.876, 0.871, 0.873]
val_acc = [0.875, 0.863, 0.876]
xlabel = 'Activation Function'
ylabel = 'Accuracy'
title = 'Neural Network with Varying Activation Function'
plotModels(activation_fn, train_acc, val_acc, xlabel, ylabel, title)



