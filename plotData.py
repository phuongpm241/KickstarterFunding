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

def plotConvergence(x, train_acc, xlabel, ylabel, title, xticks):
    plt.scatter(x, train_acc, c = 'blue', marker = '.')
    plt.plot(x, train_acc, 'b--', label = 'Conv epochs')
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    min_of_min = min(train_acc)
    max_of_max = max(train_acc)

    min_val = min_of_min - (max_of_max - min_of_min)*0.1
    max_val = max_of_max + (max_of_max - min_of_min)*0.1
    plt.ylim(min_val, max_val)
    if xticks != None:
        plt.xticks(x, xticks)
    plt.title(title)
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


# lambda_val_tick = ['2e-8', '2e-6', '2e-4', '0.02', '0.2', '2', '20', '2e2', '2e4']
# lambda_val = [-8,-6,-4,-2,-1,0,1,2,4]
# train_acc = [0.652, 0.652, 0.652, 0.668, 0.748, 0.760, 0.763, 0.801, 0.559]
# val_acc = [0.652, 0.652, 0.652, 0.666, 0.747, 0.760, 0.763, 0.796, 0.557]
# xlabel = 'Regularization Parameter'
# ylabel = 'Accuracy'
# title = 'SVM with Varying Regularization Parameter'
# #plotModels(lambda_val, train_acc, val_acc, xlabel, ylabel, title)
# plotLine(lambda_val, train_acc, val_acc, xlabel, ylabel, title, lambda_val_tick)

# iterations = [5, 10, 25, 50, 100]
# train_acc = [0.761, 0.762, 0.763, 0.763, 0.764]
# val_acc = [0.760, 0.761, 0.763, 0.763, 0.763]

# iterations = np.arange(1, 101)
# train_acc = [0.7122874351178572, 0.7122527542397373, 0.7123683571668035, 0.712472399801163, 0.712530201264696, 0.7125648821428159, 0.7125764424355224, 0.7125995630209357, 0.7126573644844687, 0.7127267262407084, 0.7127498468261216, 0.7128192085823613, 0.7128307688750679, 0.7128654497531878, 0.7129232512167208, 0.7129694923875473, 0.7130272938510803, 0.7130388541437869, 0.7130735350219067, 0.7130966556073199, 0.7131891379489729, 0.7131891379489729, 0.7132584997052125, 0.7132816202906258, 0.7136053084864109, 0.7136168687791175, 0.7136284290718241, 0.7136399893645307, 0.7136515496572373, 0.7136746702426505, 0.7140677201946753, 0.7140677201946753, 0.7141024010727952, 0.7141139613655018, 0.7141833231217415, 0.7141833231217415, 0.714194883414448, 0.7142064437071547, 0.714345167219634, 0.714345167219634, 0.7143682878050472, 0.7143682878050472, 0.7143682878050472, 0.7144376495612869, 0.7144376495612869, 0.7144376495612869, 0.7144607701467001, 0.7144607701467001, 0.7144607701467001, 0.7144723304394067, 0.7144723304394067, 0.7144838907321134, 0.7144954510248199, 0.7144954510248199, 0.7144954510248199, 0.7144954510248199, 0.7144954510248199, 0.7144954510248199, 0.7145185716102331, 0.7145185716102331, 0.7145185716102331, 0.7145185716102331, 0.7145185716102331, 0.7145185716102331, 0.7145301319029398, 0.714553252488353, 0.714553252488353, 0.7145648127810597, 0.7145763730737662, 0.7145763730737662, 0.7145763730737662, 0.7145763730737662, 0.7145763730737662, 0.7146457348300059, 0.7146457348300059, 0.7146572951227125, 0.7146804157081257, 0.7146804157081257, 0.7146804157081257, 0.7146919760008323, 0.7146804157081257, 0.7146804157081257, 0.7146804157081257, 0.7146804157081257, 0.7146804157081257, 0.714703536293539, 0.7147150965862455, 0.7147150965862455, 0.7147150965862455, 0.7147150965862455, 0.7147150965862455, 0.7147150965862455, 0.7147150965862455, 0.7148422598060183, 0.7148422598060183, 0.7148422598060183, 0.7148422598060183, 0.7148538200987249, 0.7148653803914315, 0.7148653803914315]
# val_acc = [0.7122445204846019, 0.7121520392120596, 0.712290761120873, 0.7128456487561269, 0.7128456487561269, 0.712891889392398, 0.7129843706649404, 0.7129843706649404, 0.7129843706649404, 0.7130306113012115, 0.7130306113012115, 0.713169333210025, 0.7132155738462961, 0.7132618144825673, 0.7133080551188384, 0.7133080551188384, 0.7132618144825673, 0.7132618144825673, 0.7132618144825673, 0.7132618144825673, 0.7132618144825673, 0.7133080551188384, 0.713493017663923, 0.713493017663923, 0.7136317395727365, 0.7135854989364654, 0.7135854989364654, 0.7136317395727365, 0.7136317395727365, 0.7136779802090076, 0.7140016646629057, 0.7140479052991769, 0.7141403865717192, 0.7141403865717192, 0.7141403865717192, 0.7141403865717192, 0.7141403865717192, 0.7141403865717192, 0.7142791084805327, 0.7142791084805327, 0.7142791084805327, 0.7142791084805327, 0.7142791084805327, 0.7144178303893461, 0.7144178303893461, 0.7144178303893461, 0.7144178303893461, 0.7144178303893461, 0.7144178303893461, 0.7144178303893461, 0.7144178303893461, 0.7144178303893461, 0.7144178303893461, 0.7144178303893461, 0.7144178303893461, 0.7144178303893461, 0.7145103116618885, 0.7145103116618885, 0.7145103116618885, 0.7145103116618885, 0.7145103116618885, 0.7145103116618885, 0.7145103116618885, 0.7145103116618885, 0.7145103116618885, 0.7145103116618885, 0.7145103116618885, 0.7145103116618885, 0.7144640710256173, 0.7145103116618885, 0.7145103116618885, 0.7145103116618885, 0.7145103116618885, 0.7145565522981596, 0.7145565522981596, 0.7145565522981596, 0.7145565522981596, 0.7146027929344307, 0.7146027929344307, 0.7146027929344307, 0.7146027929344307, 0.7146027929344307, 0.7146027929344307, 0.7146027929344307, 0.7146027929344307, 0.7146027929344307, 0.714649033570702, 0.714649033570702, 0.714649033570702, 0.714649033570702, 0.714649033570702, 0.714649033570702, 0.714649033570702, 0.7147877554795155, 0.7147877554795155, 0.7147877554795155, 0.7147877554795155, 0.7148802367520577, 0.7148802367520577, 0.7148802367520577]

# iterations = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# train_acc = [train_acc[9], train_acc[19], train_acc[29], train_acc[39], train_acc[49],
#             train_acc[59], train_acc[69], train_acc[79], train_acc[89], train_acc[99]]
# val_acc = [val_acc[9], val_acc[19], val_acc[29], val_acc[39], val_acc[49],
#             val_acc[59], val_acc[69], val_acc[79], val_acc[89], val_acc[99]]

# xlabel = 'Iterations'
# ylabel = 'Accuracy'
# title = 'SVM with Varying Iterations'
# #plotParameters(iterations, train_acc, val_acc, xlabel, ylabel, title)
# plotLine(iterations, train_acc, val_acc, xlabel, ylabel, title, None)

# lambda_val = [600, 750, 790, 800, 810, 850, 900]
# conv = [1, 1, 61, 220, 699, 1000, 1000]
# xlabel = 'Regularization Parameter'
# ylabel = 'Convergence Iterations'
# title = 'SVM Convergence with Varying Regularization Parameter'
# #plotModels(lambda_val, train_acc, val_acc, xlabel, ylabel, title)
# plotConvergence(lambda_val, conv, xlabel, ylabel, title, None)


# ##############################
# ##							##
# ##		  KERNEL SVM		##
# ##							##
# ##############################

# epochs_tick = ['1e0', '1e1', '1e2', '1e3', '1e4', '1e5', '1e6', '1e7']
# epochs = [0, 1, 2, 3, 4, 5, 6, 7]
# train_acc = [0.659, 0.311, 0.304, 0.307, 0.316, 0.736, 0.751, 0.763]
# val_acc = [0.657, 0.311, 0.305, 0.308, 0.318, 0.737, 0.750, 0.761]
# xlabel = 'Epochs (log scale)'
# ylabel = 'Accuracy'
# title = 'Linear Kernel SVM with Varying Training Epochs'
# plotLine(epochs, train_acc, val_acc, xlabel, ylabel, title, epochs_tick)

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
# gamma_val_tick = ['2e-10', '2e-5', '0.02', '0.2', '2', '20','2e5']
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

# regularizer = ['liblinear', 'lbgfs', 'saga', 'sag']
# x = [1, 2, 3, 4]
# train_acc = [0.7841, 0.7860, 0.6760, 0.7416]
# val_acc = [0.7842, 0.7866, 0.6760, 0.7393]
# xlabel = 'Solver'
# ylabel = 'Accuracy'
# title = 'Logistic Regression with Varying Solver'
# plotModels(regularizer, train_acc, val_acc, xlabel, ylabel, title)
# #plotLine(x, train_acc, val_acc, xlabel, ylabel, title, regularizer)

# C = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]
# x = [-6, -5, -4, -3, -2, -1, 0, 1]
# train_acc = [0.7579, 0.7668, 0.7803, 0.7855, 0.7859, 0.7861, 0.7861, 0.7861]
# val_acc = [0.7576, 0.7652, 0.7800, 0.7851, 0.7862, 0.7866, 0.7866, 0.7866] 
# xlabel = 'Regularization Parameter (log scale)'
# ylabel = 'Accuracy'
# title = 'Logistic Regression with Varying Regularization Parameter'
# #plotParameters(C, train_acc, val_acc, xlabel, ylabel, title)
# plotLine(x, train_acc, val_acc, xlabel, ylabel, title, C)

# iterations = [1, 10, 50, 100]
# train_acc = [0.4385, 0.7685, 0.7861, 0.7861]
# val_acc = [0.4373, 0.7682, 0.7866, 0.7866]
# xlabel = 'Iterations'
# ylabel = 'Accuracy'
# title = 'Logistic Regression with Varying Iterations'
# #plotParameters(iterations, train_acc, val_acc, xlabel, ylabel, title)
# plotLine(iterations, train_acc, val_acc, xlabel, ylabel, title, None)

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

# 3 Hidden Layer, ReLU, Random Init, 100 Neurons, 50 Epochs, 0.05 Dropout
# Optimizer 
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



