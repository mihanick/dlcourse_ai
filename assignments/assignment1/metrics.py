import numpy as np

def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    
    num_samples = prediction.shape[0]
    
    true_positives = np.sum(prediction & ground_truth)
    true_negatives = np.sum((~prediction) & (~ground_truth))
    false_positives = np.sum(prediction & (~ground_truth))
    false_negatives = np.sum((~prediction) & ground_truth)
    
    accuracy = (true_positives + true_negatives) / num_samples
    
    recall =  true_positives /(true_positives + false_negatives)
    precision = true_positives / (true_positives + false_positives)
    
    f1 = (1+1)*(precision * recall)/(1*precision + recall)
    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    
    # TODO: Implement computing accuracy
    
    num_samples = prediction.shape[0]
    correct = 0
    for i in range(num_samples):
        if prediction[i] == ground_truth[i]:
            correct+=1
    
    return correct/num_samples
