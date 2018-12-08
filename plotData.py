import matplotlib.pyplot as plt
import pylab as pl
import numpy as np 

def plotModels(x, train_acc, val_acc, xlabel, ylabel, title):
    min_of_min = min(min(train_acc), min(val_acc))
    max_of_max = max(max(train_acc), max(val_acc))
    min_val = min_of_min - (max_of_max - min_of_min)*0.1
    max_val = max_of_max + (max_of_max - min_of_min)*0.1
    plt.ylim(min_val, max_val)
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

def plotLine(x, train_acc, val_acc, xlabel, ylabel, title, xticks):
    plt.scatter(x, train_acc, c = 'blue', marker = '.')
    plt.plot(x, train_acc, 'b--', label = 'Train acc')
    plt.scatter(x, val_acc, c = 'red', alpha = 1, marker = '.')
    plt.plot(x, val_acc, 'r--', label ='Val acc', alpha = 0.7)
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    min_of_min = min(min(train_acc), min(val_acc))
    max_of_max = max(max(train_acc), max(val_acc))

    min_val = min_of_min - (max_of_max - min_of_min)*0.1
    max_val = max_of_max + (max_of_max - min_of_min)*0.1
    plt.ylim(min_val, max_val)
    if xticks != None:
        plt.xticks(x, xticks)
    plt.title(title)
    plt.show()

def plotParameters(x, train_acc, val_acc, xlabel, ylabel, title):
	plt.plot(x, train_acc, 'g-', label='Training Accuracy')
	plt.plot(x, val_acc, 'b-', label='Validation Accuracy')
	plt.ylabel(ylabel)
	plt.xlabel(xlabel)
	plt.title(title)
	plt.legend(loc='best')
	plt.show()

# ##############################
# ##							##
# ##		 PEGASOS SVM		##
# ##							##
# ##############################


# lambda_val_tick = ['2e-10', '2e-8', '2e-6', '2e-4', '2e-2', '2e-1', '2e0', '2e1', '2e2', '2e4']
# lambda_val = [-10,-8,-6,-4,-2,-1,0,1,2,4]
# train_acc = [0.652, 0.652, 0.652, 0.652, 0.668, 0.748, 0.760, 0.763, 0.801, 0.559]
# val_acc = [0.652, 0.652, 0.652, 0.652, 0.666, 0.747, 0.760, 0.763, 0.796, 0.557]
# xlabel = 'Regularization Parameter'
# ylabel = 'Accuracy'
# title = 'SVM with Varying Regularization Parameter'
# #plotModels(lambda_val, train_acc, val_acc, xlabel, ylabel, title)
# plotLine(lambda_val, train_acc, val_acc, xlabel, ylabel, title, lambda_val_tick)

# iterations = [5, 10, 25, 50, 100]
# train_acc = [0.802, 0.802, 0.802, 0.802, 0.801]
# val_acc = [0.797, 0.797, 0.797, 0.797, 0.797]
# xlabel = 'Iterations'
# ylabel = 'Accuracy'
# title = 'SVM with Varying Iterations'
# #plotParameters(iterations, train_acc, val_acc, xlabel, ylabel, title)
# plotLine(iterations, train_acc, val_acc, xlabel, ylabel, title, None)

# ##############################
# ##							##
# ##		  KERNEL SVM		##
# ##							##
# ##############################

# # Polynomial Kernel
# # fixed coefficient = 1
# degrees = [1, 2, 4, 8, 10]
# train_acc = [0.680, 0.764, 0.681, 0.680, 0.320]
# val_acc = [0.680, 0.762, 0.680, 0.680, 0.320]
# xlabel = 'Degree'
# ylabel = 'Accuracy'
# title = 'Polynomial Kernel SVM with Varying Degree'
# plotLine(degrees, train_acc, val_acc, xlabel, ylabel, title, None)

# # fix degree = 2
# coeffs = [1e-1, 1, 5, 10, 25, 50, 100]
# train_acc = [0.298,0.764,0.704,0.716,0.733,0.717,0.296]
# val_acc = [0.298,0.762,0.704,0.718,0.734,0.719,0.296]
# xlabel = 'Coefficient'
# ylabel = 'Accuracy'
# title = 'Polynomial Kernel SVM with Varying Coefficient'
# plotLine(coeffs, train_acc, val_acc, xlabel, ylabel, title, None)



# # Gaussian kernel
# gamma_val_tick = ['2e-10', '2e-5', '2e-2', '2e-1', '2e0', '2e0', '1e1','1e5']
# gamma_val = [-10, -5, -2, -1, 0, 1, 5]
# train_acc = [0.692,  0.821, 0.875, 0.882, 0.911, 0.957, 0.994]
# val_acc = [0.691, 0.818, 0.872, 0.873, 0.860, 0.768, 0.686]
# xlabel = 'Gamma Value'
# ylabel = 'Accuracy'
# title = 'Gaussian Kernel SVM with Varying Gamma Value'
# plotLine(gamma_val, train_acc, val_acc, xlabel, ylabel, title, gamma_val_tick)

##############################
##							##
##	  LOGISTIC REGRESSION	##
##							##
##############################

regularizer = ['liblinear', 'lbgfs', 'saga', 'sag']
x = [1, 2, 3, 4]
train_acc = [0.7841, 0.7860, 0.6760, 0.7416]
val_acc = [0.7842, 0.7866, 0.6760, 0.7393]
xlabel = 'Solver'
ylabel = 'Accuracy'
title = 'Logistic Regression with Varying Solver'
plotModels(regularizer, train_acc, val_acc, xlabel, ylabel, title)
#plotLine(x, train_acc, val_acc, xlabel, ylabel, title, regularizer)

C = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]
x = [-6, -5, -4, -3, -2, -1, 0, 1]
train_acc = [0.7579, 0.7668, 0.7803, 0.7855, 0.7859, 0.7861, 0.7861, 0.7861]
val_acc = [0.7576, 0.7652, 0.7800, 0.7851, 0.7862, 0.7866, 0.7866, 0.7866] 
xlabel = 'Regularization Parameter (log scale)'
ylabel = 'Accuracy'
title = 'Logistic Regression with Varying Regularization Parameter'
#plotParameters(C, train_acc, val_acc, xlabel, ylabel, title)
plotLine(x, train_acc, val_acc, xlabel, ylabel, title, C)

iterations = [1, 10, 50, 100]
train_acc = [0.4385, 0.7685, 0.7861, 0.7861]
val_acc = [0.4373, 0.7682, 0.7866, 0.7866]
xlabel = 'Iterations'
ylabel = 'Accuracy'
title = 'Logistic Regression with Varying Iterations'
#plotParameters(iterations, train_acc, val_acc, xlabel, ylabel, title)
plotLine(iterations, train_acc, val_acc, xlabel, ylabel, title, None)

# ##############################
# ##							##
# ##	    NEURAL NETWORKS		##
# ##							##
# ##############################

# # 3 Hidden Layer, Adamax, ReLU, Random Init, 0.1 Dropout, 50 Epochs
# # Varying Neurons 

# # Input -> 100 -> 50 -> 10 -> Output
# # Input -> 1000 -> 50 -> 10 -> Output 
# # Input -> 5000 -> 50 -> 10 -> Output 

# units = [100, 1000, 5000]
# train_acc = [0.875, 0.875, 0.875]
# val_acc = [0.876, 0.871, 0.872]
# xlabel = 'Hidden Units'
# ylabel = 'Accuracy'
# title = 'Neural Network with Varying Hidden Units'
# #plotParameters(units, train_acc, val_acc, xlabel, ylabel, title)
# plotLine(units, train_acc, val_acc, xlabel, ylabel, title, None)

# # Input -> 100 -> 50 -> 10 -> Output
# # Input -> 1000 -> 500 -> 100 -> Output 
# # Input -> 5000 -> 2500 -> 500 -> Output 

# units = [100, 1000, 5000]
# train_acc = [0.875, 0.875, 0.686]
# val_acc = [0.876, 0.872, 0.682]
# xlabel = 'Hidden Units'
# ylabel = 'Accuracy'
# title = 'Neural Network with Varying Hidden Units'
# plotLine(units, train_acc, val_acc, xlabel, ylabel, title, None)

# # 3 Hidden Layer, Adamax, ReLU, Random Init, 100 Neurons, 50 Epochs
# # Dropout 
# dropout = [0, 0.05, 0.1, 0.2, 0.3, 0.5]
# train_acc = [0.876, 0.876, 0.875, 0.874, 0.872, 0.861]
# val_acc = [0.876, 0.875, 0.876, 0.876, 0.877, 0.868]
# xlabel = 'Dropout Probability'
# ylabel = 'Accuracy'
# title = 'Neural Network with Varying Dropout Probability'
# plotLine(dropout, train_acc, val_acc, xlabel, ylabel, title, None)

# # 3 Hidden Layer, ReLU, Random Init, 100 Neurons, 50 Epochs, 0.05 Dropout
# # Optimizer 
# optimizer = ['Adam', 'Adamax', 'Adagrad', 'SGD', 'Adadelta']
# train_acc = [0.875, 0.876, 0.876, 0.858, 0.875]
# val_acc = [0.876, 0.876, 0.877, 0.874, 0.874]
# xlabel = 'Optimizer'
# ylabel = 'Accuracy'
# title = 'Neural Network with Varying Optimizer'
# plotModels(optimizer, train_acc, val_acc, xlabel, ylabel, title)

# # Adamax, ReLU, Random Init, 100 Neurons, 0.05 Dropout, 50 Epochs
# # Hidden Layer 

# # Input -> 100 -> 50 -> 10 -> Output 
# # Input -> 200 -> 100 -> 50 -> 25 -> 10 -> Output 
# # Input -> 500 -> 250 -> 150 -> 100 -> 50 -> 25 -> 10 -> Output

# hidden_layer = [3, 5, 7]
# train_acc = [0.876, 0.876, 0.875]
# val_acc = [0.875, 0.878, 0.878]
# xlabel = 'Hidden Layers'
# ylabel = 'Accuracy'
# title = 'Neural Network with Varying Hidden Layers'
# plotLine(hidden_layer, train_acc, val_acc, xlabel, ylabel, title, hidden_layer)

# # 3 Hidden Layer, Adagrad, ReLU, Random Init, 100 Neurons, 50 Epochs, 0.05 Dropout
# # Batch Normalization 
# normalization = ['false', 'true']
# train_acc = [0.876, 0.870]
# val_acc = [0.875, 0.576]
# xlabel = 'Batch Normalization'
# ylabel = 'Accuracy'
# title = 'Neural Network with/without Batch Normalization'
# plotModels(normalization, train_acc, val_acc, xlabel, ylabel, title)

# # 3 Hidden Layer, Adagrad Random Init, 100 Neurons, 50 Epochs, 0.05 Dropout
# activation_fn = ['relu', 'tanh', 'sigmoid']
# train_acc = [0.876, 0.871, 0.873]
# val_acc = [0.875, 0.863, 0.876]
# xlabel = 'Activation Function'
# ylabel = 'Accuracy'
# title = 'Neural Network with Varying Activation Function'
# plotModels(activation_fn, train_acc, val_acc, xlabel, ylabel, title)



