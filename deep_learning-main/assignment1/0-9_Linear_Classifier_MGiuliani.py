'''
prompt: inputs: 256 length vectors, weights: 10x256, output vector length 10. simple loop using cross entropy loss to update the weights of this classifier

To implement a simple classifier with a cross-entropy loss and update the weights through a loop, we can create a model where:

Inputs are 256-length vectors.
Weights are a 10x256 matrix (for 10 classes, each class having a weight vector of length 256).
Output is a vector of length 10, representing the probabilities of each class.

multi-class logistic regression classifier.
- Compute the probabilities using the softmax function, which generalizes logistic regression to multiple classes.
- The softmax function converts the raw class scores (logits) into probabilities by comparing the exponentiated logits of each class against the sum of exponentiated logits for all classes.

Here’s a basic outline of what we'll implement:

- Initialize the weights.
- Loop over the dataset:
  - Compute the logits.
  - Apply the softmax function to get probabilities.
  - Compute the cross-entropy loss.
  - Compute the gradient of the loss with respect to the weights.
  - Update the weights using gradient descent.
  - Iterate until convergence or for a fixed number of epochs.
'''

import numpy as np
import cv2
np.set_printoptions(suppress=True) # override annoying numpy print formatting



"""
Summary: Initializes weights for training and reads in the training data and labels.
Inputs:  num_classes - Number of outputs in the linear classifier.
         num_features - Number of features in each input (pixels for this linear classifier).
Returns: weights - Initialized small random weights.
         training_data - Full training dataset 1000x256.
         training_labels - Ground truth labels for each training input 1000x10.
"""
def training_init(num_classes, num_features):
    """
    This creates the array of weights from each perceptron to each output (256x10). We want the weights to be small, random, and non-zero, 
    so we generate random standard normally distributed weights (bell curve, mean 0, std dev. 1) and multiply them all by 0.01 to make them small.
    We want the weights to be small to avoid overfitting, underfitting, local minima, and local maxima.
    """
    weights = 0.01 * np.random.randn(num_classes, num_features)

    """
    Read in training data and transpose it from 256x1000 to 1000x256 so each main index is an image.
    Input data file formatted as each row is a pixel and each column is an image.
    We want each row to be an image for the way this code is written.
    """
    training_data = np.genfromtxt(r'data\\training_inputs_256x1000.csv', dtype=np.float64, delimiter=',').transpose()

    """
    Read in the training labels and transpose them from 10x1000 to 1000x10 so each main index is the one-hot encoded ground truth for each input
    Training label file is formatted as each row is the one-hot formatted label for a potential output, and each column is the full one-hot encoded label for an input
    We want each row to be the one-hot encoded label for an input for the way this code is written.
    """
    training_labels = np.genfromtxt(r'data\\training_targets_10x1000_0or1.csv', dtype=np.float64, delimiter=',').transpose()
    
    return weights, training_data, training_labels

"""
Summary: Takes the raw prediction values for each potential output of a given input and applies the softmax equation to them...
         ...to produce a percentage probability (0.00 - 1.00) that a given input matches each output based on that raw prediction.
         Euler's number, e, to the power of the given class, divided by the sum of e to the power of each respective logit value 
         (e^li)/(e^l0 + e^l1 + e^l2 + e^l3 + e^l4 + e^l5 + e^l6 + e^l7 + e^l8 + e^l9) gives us the probability for each prediction.
Inputs:  logits - Raw class prediction values from each input in a batch
Returns: Probability percentage predictions for each class for each input in the batch. Sum of predictions for each input should equal 1.00.
"""
def softmax(logits):
    exps = np.exp(logits - np.max(logits, axis=1, keepdims=True)) # each value of this array is Euler's number e to the power of each logit value
    
    # Divides the value at each index of the array produced by the line above by the sum of all indices of the array produced by the line above. 
    # Result is the probability (0.00 - 1.00) that each prediction matches the ground truth output for the given input. Sum of all indices of returned array should equal 1.00 for each input.
    return exps / np.sum(exps, axis=1, keepdims=True)


"""
Summary: Calculate the cross entropy loss of each input batch as a quantification of the performance of the classifier for this batch.
Inputs:  y_true - The one-hot encoded ground truth labels of the training data in the batch (the correct/intended output for each input)
         probabilities - The percentage probability output guesses for the predictions for each piece of training data in the batch (the predicted output for each input)
Returns: Average cross entropy loss for the batch.
"""
def cross_entropy_loss(y_true, probabilities):
    """
    The argument to np.log indexes into the probabilities array to get a prediction value for each input.
    range(len(probabilities)) selects the index for all training inputs.
    y_true.argmax(axis=1) gets the prediction value that each input made for the correct class by...
      ...selecting the index of each input's predictions that corresponds to the 1 in its one-hot encoded label.
    Therefore, we are passing an array to -np.log where each index is the prediction made by each input for its correct class.
    
    We take the natural log to get a loss value where numbers closer to 1 give a small value and numbers further than 1 give a larger value.
    Basically, the worse the prediction, the greater the loss.
    We flip the sign since we are minimizing.
    """
    log_probs = -np.log(probabilities[range(len(probabilities)), y_true.argmax(axis=1)])
    
    # We return the mean (average) loss of the batch to give us one digestible number to quantify the loss of this batch
    return np.mean(log_probs)


"""
Summary: Calculates the average gradient for each feature (pixel) of the inputs.
Inputs:  input_batch - Batch of inputs
         label_batch - One-hot encoded ground truth labels for each input of the batch
         probabilities - Probability prediction percentages by each input for how likely it is that the input matches a possible output 0 - 9
Returns: The average gradient value for each feature (pixel) of the inputs.
"""
def compute_gradients(input_batch, label_batch, probabilities):
    """
    We subtract the one-hot encoded ground truth labels from the predictions to get how far off each prediction was from its perfect correct predictions (the ground truth).
    So each prediction value for the incorrect classes remain the same, because they should be zero, and whatever number they are is how far off of zero they are.
    The prediction for the correct class for each input has 1 subtracted from it, so if the prediction is 0.78, the error is -0.22, because the prediction was 0.22 under what it should be.
    """
    errors = probabilities - label_batch

    """
    Here, we perform a dot product of the transposed input batch and the errors. This quantifies how much each input feature effects the error.
    We divide by the number of inputs in the batch to get the average gradient for the batch so we can adjust the weights.
    This tells us how much and in what direction we need to adjust the weights for each feature.
    """
    return np.dot(input_batch.T, errors) / len(input_batch)


"""
Summary: Uses the gradients, modified by the learning rate, to adjust the weights after processing each input batch.
Inputs:  weights - Weights control how much impact each individual input feature (each pixel in this case) has on the output.
         gradients - Gradients to apply to each neuron. These tell us whether to increase or decrease a given weight and by how much.
         learning rate - Controls how much a weight is allowed to change in one pass. This helps to avoid overshooting, undershooting, and local minima/maxima.
Returns: The updated array of weights for each input feature (pixel).
"""
def update_weights(weights, gradients, learning_rate):
    """
    We multiply the gradients by the learning rate to adjust how big of an adjustment the gradients are allowed to make on the weights in on training pass.
    The learning rate should be tuned to avoid overshooting, undershooting, and local minima/maxima.
    Once we adjust the gradients with the learning rate, we subtract them from the weights to obtain our new and improved weights.
    """
    return weights - (learning_rate * gradients) # added parentheses around learning_rate * gradients according to order of operations to improve readability and clarity


"""
Summary: Trains the linear classifier.
Inputs:  training_data - Full dataset of training images.
         training_labels - One-hot encoded ground truth labels for the training dataset.
         weights - Random small weights to begin training with.
         epochs - Number of times the entire training dataset will be ran through the linear classifier to adjust the weights.
         batch_size - Number of inputs to compute average loss and gradients before adjusting weights.
         learning_rate - Value that limits the size of the gradients to control how much they can change the weights in one batch.
Returns: weights - Final weights after training is complete.
         loss - Final cross entropy loss value to quantify the performance of the linear classifier on the training dataset.
"""
def train(training_data, training_labels, weights, epochs, batch_size, learning_rate):
    # Training loop
    for epoch in range(epochs): # one loop = one full training pass of all of the training data
        for i in range(0, len(training_data), batch_size): # breaks the training dataset into batches. i starts at 0 and increases by batch_size each loop.
            training_data_batch = training_data[i:i + batch_size] # copies only the inputs of the current batch from the master dataset into the local training batch dataset
            training_labels_batch = training_labels[i:i + batch_size]  # copies the class labels of the current batch from the master class label dataset into the local batch dataset
            
            logits = np.dot(training_data_batch, weights.T) # dot product of the inputs with their weights produces a raw prediction value for each class which we refer to as a logit
            
            probabilities = softmax(logits) # runs each logit through the softmax equation, which converts the raw predictions into a percentage probability that the input matches a given output
            
            loss = cross_entropy_loss(training_labels_batch, probabilities) # use the ground truth labels and probability guesses to calculate cross entropy loss for this pass
            
            gradients = compute_gradients(training_data_batch, training_labels_batch, probabilities) # computes how much and in what direction we need to change the weights for each feature to reduce the error of the predictions
            
            weights = update_weights(weights, gradients.T, learning_rate) # now that the training data is calculated for the batch, us it to adjust the weights to reduce loss and prediction error
            
        if epoch % 50 == 0: # Every multiple of 50 epochs, we print which epoch we are on and what our loss is to the console
            print(f"Epoch {epoch}, Loss: {loss}")
    
    save_visual_weights(weights.transpose()) # save the weights as PNG images to visualize them. Result is what looks like a mask for each output number. Transpose the weights so images are created correctly.

    print("\r\nFinal Training Loss: " + str(loss)) # Output the final weights and loss

    return weights, loss


"""
Summary: Takes the final trained weights and turns them into PNG images so we can visualize what the classifier thinks each number should look like.
Inputs:  transposed_final_weights - Weights calculated by the full training process. Weights should be transposed when passed so the image is constructed correctly.
Outputs: Ten 16x16 pixel PNG images, each visualizing the weights of the ten outputs 0-9. Each image represents what the classifier thinks each number looks like.
Returns: None
"""
def save_visual_weights(transposed_final_weights):
    for i in range(0, 10): # for each output number 0-9
        weights_image = transposed_final_weights[:,i] # Get all 256 weights for each weight
        
        weights_img_array = np.resize(weights_image, (16, 16)) # reshape the weights from a flat array to a 16x16 array.
        
        weights_img_array *= 255.0 # change the brightness values 0-1 for each pixel to 8-bit brightness values 0-255
        
        img_directory = r'data\\training_weights_images\\' # relative path to the folder storing the images
        img_filename = r'img' + str(i) + r'.png' # concatenate file name and type. img0.png = output mask for 0.
        img_path = img_directory + img_filename # concatenate full file path

        cv2.imwrite(img_path, weights_img_array) # write weight mask data as an image to the specified path
    return


"""
Summary: Validates the linear classifier by letting us evaluate the perforance with known inputs and expected outputs that do not affect the weights.
Inputs:  weights - Fully trained weights to be validated.
Returns: loss - Final cross entropy loss value to quantify the performance of the linear classifier on the validation dataset.
"""
def validate(weights):
    validation_data = np.genfromtxt(r'data\\validation_inputs_256x1000.csv', dtype=np.float64, delimiter=',').transpose() # Read in validation dataset
    validation_labels = np.genfromtxt(r'data\\validation_targets_10x1000_0or1.csv', dtype=np.float64, delimiter=',').transpose() # Read in labels for the validation dataset
    
    logits = np.dot(validation_data, weights.T) # dot product of the inputs with their weights produces a raw prediction value for each class which we refer to as a logit  
    probabilities = softmax(logits) # runs each logit through the softmax equation, which converts the raw predictions into a percentage probability that the input matches a given output 
    
    loss = cross_entropy_loss(validation_labels, probabilities) # use the ground truth labels and probability guesses to calculate cross entropy loss

    print()
    print("Example Validation Prediction:      " + str(probabilities[0]))
    print("Target for Validation Prediction:   " + str(validation_labels[0]))
    print("Validation Loss:                    " +  str(loss))

    return loss # return the loss so it can be used to evaluate the performance


"""
Summary: Tests the trained weights of the linear classifier on a large dataset so we can evaluate performance in a large scope.
Inputs:  weights - Fully trained weights to be tested.
Returns: loss - Final cross entropy loss value to quantify the performance of the linear classifier on the test dataset.
"""
def test(weights):
    test_data = np.genfromtxt(r'data\\test_inputs_256x9000.csv', dtype=np.float64, delimiter=',').transpose() # Read in validation dataset
    test_labels = np.genfromtxt(r'data\\test_targets_10x9000_0or1.csv', dtype=np.float64, delimiter=',').transpose() # Read in labels for the validation dataset
    
    logits = np.dot(test_data, weights.T) # dot product of the inputs with their weights produces a raw prediction value for each class which we refer to as a logit  
    probabilities = softmax(logits) # runs each logit through the softmax equation, which converts the raw predictions into a percentage probability that the input matches a given output 
    
    loss = cross_entropy_loss(test_labels, probabilities) # use the ground truth labels and probability guesses to calculate cross entropy loss

    print()
    print("Example Test Prediction:      " + str(probabilities[0]))
    print("Target for Test Prediction:   " + str(test_labels[0]))
    print("Testing Loss:                 " +  str(loss))

    return loss


def main():
    num_features = 256      # Input is 16 x 16 black and white image. 256 pixels, each pixel is a feature. Each fixture is a brightness value 0 to 1.000000000. 9 decimal places.
    num_classes = 10        # Output is the probability of the input image being a given number 0 - 9, so 10 classes total
    
    learning_rate = 0.01    # Controls how much each weight is allowed to change in one pass. Tuning this value is important to optimize training speed and results.
    epochs = 1000           # The number of training passes of the full training dataset we will make on the classifier before validation and testing
    batch_size = 32         # This is the number of peices of input data we will process before each adjustment of our weights


    weights, training_data, training_labels = training_init(num_classes, num_features) # initialize small random weights and read in the training data
    weights, loss = train(training_data, training_labels, weights, epochs, batch_size, learning_rate) # train on training dataset using the parameters set above

    validate(weights) # Run known inputs and with ground truth labels through the linear classifier and calculate the cross entropy loss to validate performance of our weights.

    test(weights)

    return

if __name__ == "__main__":
    main()
