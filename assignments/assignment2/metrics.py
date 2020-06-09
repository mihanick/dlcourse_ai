def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    print(prediction, ground_truth)
    # TODO: Implement computing accuracy
    
    num_samples = prediction.shape[0]
    correct = 0
    for i in range(num_samples):
        if prediction[i] == ground_truth[i]:
            correct+=1
    
    return correct/num_samples
