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
    
    Why initialize weights like this?
    - If the weights are zero, the outputs will be zero and no learning will occur (Input * 0 = 0).
    - If any given weight is significantly larger or smaller than the others, that weight will have a skewed affect on learning (e.g. disproportionate influence on outputs & gradients).
    - Randomization breaks symmetry, so several neurons do not have the same behavior unintentionally.
    - Since small networks like linear classifiers don't involve backpropagation, which uses the chain rule to derive the weights for deeper layers, initializing weights like we do here is okay.
       - In deeper networks, weights being too small can cause them to vanish as the chain rule will result in small weights multiplying together and getting smaller as we go deeper.
       - Likewise, large weights will cause activation functions to saturate as we go deeper, and learning becomes very unstable as they cause weights to change too much. 
       - Deeper networks can use methods like Xavier Initialization to scale the initial weights to the size of the network so they do not vanish or explode.
    - Like most things, there is no one best way to initialize weights for all models.
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
Summary: Reads in the validation data and labels.
Inputs:  
Returns: validation_data - Full validation dataset 1000x256.
         validation_labels - Ground truth labels for each validation input 1000x10.
"""
def validation_init():
    validation_data = np.genfromtxt(r'data\\validation_inputs_256x1000.csv', dtype=np.float64, delimiter=',').transpose() # Read in validation dataset
    validation_labels = np.genfromtxt(r'data\\validation_targets_10x1000_0or1.csv', dtype=np.float64, delimiter=',').transpose() # Read in labels for the validation dataset

    return validation_data, validation_labels


"""
Summary: Reads in the test data and labels.
Inputs:  
Returns: test_data - Full test dataset 1000x256.
         test_labels - Ground truth labels for each test input 1000x10.
"""
def test_init():
    test_data = np.genfromtxt(r'data\\test_inputs_256x9000.csv', dtype=np.float64, delimiter=',').transpose() # Read in test dataset
    test_labels = np.genfromtxt(r'data\\test_targets_10x9000_0or1.csv', dtype=np.float64, delimiter=',').transpose() # Read in labels for the test dataset

    return test_data, test_labels


"""
Summary: Takes the raw prediction values for each potential output of a given input and applies the softmax equation to them...
         ...to produce a percentage probability (0.00 - 1.00) that a given input matches each output based on that raw prediction.
         Euler's number, e, to the power of the given class, divided by the sum of e to the power of each respective logit value 
         (e^li)/(e^l0 + e^l1 + e^l2 + e^l3 + e^l4 + e^l5 + e^l6 + e^l7 + e^l8 + e^l9) gives us the probability for each prediction.
Inputs:  logits - Raw class prediction values from each input in a batch
Returns: Probability percentage predictions for each class for each input in the batch. Sum of predictions for each input should equal 1.00.
"""
def softmax(logits):
    """
    Why use softmax?
    - Activation functions introduce nonlinearity. The real world is not linear and activation functions allow us to more accurately model how we classify in the real world.
    - In softmax, exponentiating ensures probabilities are positive and sum to 1 whereas sigmoid just ensures each output is between 0 and 1. 
      It also emphasizes larger raw prediction values over smaller ones. This way, confident guesses have more influence, positive or negative, over the training of the model.
    - Probabilities are easier for us (humans) to interpret.
    - We calculate cross entropy loss with the probabilities.
    - Softmax is considered "smooth" because its slope does not change sharply at any point,...
      ...meaning that differentiating it will not result in large and/or unstable changes in gradients during backpropagation later on.
    - With softmax, sigmoid, and tanh, if logits are too big or small, the gradients can become extremely small or saturated, and learning will be very slow or unstable.
       - Small gradients get exponentially smaller still when differentiating during back propagation, effectively disappearing and taking learning speed to a halt.
       - Large gradients can get exponentially larger, "exploding", making learning unstable and difficult or impossible to converge.
       - Activation functions like ReLU do not saturate, which is why they are particularly popular for hidden layers.
    """
    
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
    Why use cross entropy loss?
    - Cross entropy loss gives us a mathematical method for punishing significantly incorrect class predictions much more than somewhat incorrect predictions.
    - The worse the guess, the worse the cross entropy loss is logarithmically. 95% probability for the correct class has a loss of 0.05, while 10% has a loss of 2.3.
    - This makes it clear when performance is especially poor so we can adjust hyperparameters to minimize the loss faster.
    - It also helps quantify the performance of the model by giving us one number to represent how good or bad the predictions are.
    
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
    What are gradients, what do they do, and why?
    - The gradient is a vector that points in the direction and with the magnitude of steepest increase in loss. That means to minimize loss, we want to subtract the gradients from the weights.
    - When computing the gradient, we are calculating how much the weight of each neuron needs to be changed to reduce the error in its output for a given input. 
    - We calculate the error of each neuron's output, and calculate its dot product with the input to find how far off the weight of each neuron was from giving us the correct output. 
    - Adjusting each weight by its gradient will reduce the error in its output, but if we don't limit how big the gradients are, ... 
      ...the network could learn too fast, too slow, settle in local minima/maxima, or be too biased toward certain inputs.
    - Learning rates (implemented later when updating the weights) scale how big the gradients can be to tune how much they can affect the weights in one epoch.

    We subtract the one-hot encoded ground truth labels from the predictions to get how far off each prediction was from its perfect correct predictions (the ground truth).
    So each prediction value for the incorrect classes remain the same, because they should be zero, and whatever number they are is how far off of zero they are.
    The prediction for the correct class for each input has 1 subtracted from it, so if the prediction is 0.78, the error is -0.22, because the prediction was 0.22 under what it should be.
    """
    errors = probabilities - label_batch

    """
    Here, we perform a dot product of the transposed input batch and the errors. This tells us how much each weight needs to be adjusted to reduce the error.
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
    The gradient is a vector that points in the direction and with the magnitude of steepest increase in loss.
    That means to minimize loss, we want to subtract the gradients from the weights.
    We multiply the gradients by the learning rate to adjust how big of an adjustment the gradients are allowed to make on the weights in on training pass.
    The learning rate should be tuned to avoid overshooting, undershooting, and local minima/maxima.
    Once we adjust the gradients with the learning rate, we subtract them from the weights to obtain our new and improved weights.
    """
    return weights - (learning_rate * gradients) # added parentheses around learning_rate * gradients according to order of operations to improve readability and clarity


"""
Summary: Calculates which of the highest probabilities from each input align with their labels, then computes the total count
         of matches for the dataset and what percentage of the total were correctly predicted.
Inputs:  probabilities - The output probabilities for each input.
         one_hot_labels - The one-hot encoded ground truth labels for each input.
Returns: matches - An array for whether each input matched its label or not (0 = not matched, 1 = match)
         num_matches - Total number of inputs whose index for their highest output probability matched the true index of their one-hot encoded label.
         percent_match - Percentage of correctly guessed outputs out of the total number of inputs.
"""
def calc_performance(probabilities, one_hot_labels):
    highest_predicted_indices = np.argmax(probabilities, axis=1) # Get the index of the highest probability output for each input
    true_label_indices = np.argmax(one_hot_labels, axis=1) # get the index of the true label for each input
    matches = highest_predicted_indices == true_label_indices # check if the highest pridiction index matches the ground truth index for each input
    num_matches = np.sum(matches) # sum how many inputs predicted their label correctly
    percent_match = (num_matches / len(one_hot_labels)) * 100 # calc percentage of correct predictions

    return matches, num_matches, percent_match


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
def train(training_data, training_labels, weights, epochs, batch_size, learning_rate, validation_data, validation_labels):
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

        # We perform validation during training to evaluate how well the training is working.
        # We do this on a dataset that does not overlap with the training data so the model doesn't give a misleadingly good result by "recognizing" the input.
        validation_num_matches, validation_percent_match, validation_loss = validate(weights, validation_data, validation_labels)

        if epoch % 50 == 0: # Every multiple of 50 epochs, we print which epoch we are on and what our loss is to the console
            print(f"Epoch {epoch}")
            print(f"Validation: {validation_num_matches} / {len(validation_data)} --- {validation_percent_match:.2f}% Correct --- Loss {validation_loss}")
            print()
    
    save_visual_weights(weights.transpose()) # save the weights as PNG images to visualize them. Result is what looks like a mask for each output number. Transpose the weights so images are created correctly.

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
def validate(weights, validation_data, validation_labels):
    logits = np.dot(validation_data, weights.T) # dot product of the inputs with their weights produces a raw prediction value for each class which we refer to as a logit  
    probabilities = softmax(logits) # runs each logit through the softmax equation, which converts the raw predictions into a percentage probability that the input matches a given output 
    loss = cross_entropy_loss(validation_labels, probabilities) # use the ground truth labels and probability guesses to calculate cross entropy loss

    _, num_matches, percent_match = calc_performance(probabilities, validation_labels) # Calculate validation performace

    return num_matches, percent_match, loss # return the loss so it can be used to evaluate the performance


"""
Summary: Tests the trained weights of the linear classifier on a large dataset so we can evaluate performance in a large scope.
Inputs:  weights - Fully trained weights to be tested.
Returns: loss - Final cross entropy loss value to quantify the performance of the linear classifier on the test dataset.
"""
def test(weights, test_data, test_labels):
    logits = np.dot(test_data, weights.T) # dot product of the inputs with their weights produces a raw prediction value for each class which we refer to as a logit  
    probabilities = softmax(logits) # runs each logit through the softmax equation, which converts the raw predictions into a percentage probability that the input matches a given output 
    loss = cross_entropy_loss(test_labels, probabilities) # use the ground truth labels and probability guesses to calculate cross entropy loss

    _, num_matches, percent_match = calc_performance(probabilities, test_labels)# Calculate validation performace

    return num_matches, percent_match, loss


def main():
    num_features = 256      # Input is 16 x 16 black and white image. 256 pixels, each pixel is a feature. Each fixture is a brightness value 0 to 1.000000000. 9 decimal places.
    num_classes = 10        # Output is the probability of the input image being a given number 0 - 9, so 10 classes total
    
    # Tunable Training Hyperparameters
    learning_rate = 0.01    # Controls how much each weight is allowed to change in one pass. Tuning this value is important to optimize training speed and results.
    epochs = 1000           # The number of training passes of the full training dataset we will make on the classifier before validation and testing
    batch_size = 32         # This is the number of peices of input data we will process before each adjustment of our weights

    ################################################################################################################################################################

    weights, training_data, training_labels = training_init(num_classes, num_features) # initialize small random weights and read in the training data
    validation_data, validation_labels = validation_init()
    weights, training_loss = train(training_data, training_labels, weights, epochs, batch_size, learning_rate, validation_data, validation_labels) # train on training dataset using the parameters set above


    test_data, test_labels = test_init()
    test_num_matches, test_percent_match, test_loss = test(weights, test_data, test_labels) # Use a large dataset that the model has not seen before, along with their ground truth labels, to evaluate the final performance of the model.


    print()
    print(f"Testing Results:    {test_num_matches} / {len(test_labels)} --- {test_percent_match:.2f}% Correct --- Loss {test_loss}")
    print()

    return

if __name__ == "__main__":
    main()
