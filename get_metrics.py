from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import numpy as np 

def specificity(true, pred):
    matrix = confusion_matrix(true, pred) 
    return (matrix[0][0]/ (matrix[0][0] + matrix[0][1]))

def metrics(true, pred): 
    print ('confusion_matrix', confusion_matrix(true, pred))
    print ('precision_score: ' + str(round(precision_score(true, pred), 3)))
    print ('recall score: ' + str(round(recall_score(true, pred), 3)))
    print ('specificity: ' + str(round(specificity(true, pred), 3)))
    print ('f1 score: ' + str(round(f1_score(true, pred), 3)))