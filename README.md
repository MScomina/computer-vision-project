# Computer Vision Project
Project for the Computer Vision and Pattern Recognition course at the University of Trieste.  
The chosen project is Project 3 (CNN Classifier).  

## Project structure
The project is structured as follows, as per teacher's instructions:
### Task 1
Task 1 consists in:
- Building a CNN classifier with the following characteristics:
  - Image Input: 64x64x1
  - Convolution layer: 8 filters, 3x3, stride 1, followed by ReLU activation.
  - Max pooling layer: 2x2, stride 2
  - Convolution layer: 16 filters, 3x3, stride 1, followed by ReLU activation.
  - Max pooling layer: 2x2, stride 2
  - Convolution layer: 32 filters, 3x3, stride 1, followed by ReLU activation.
  - Fully connected layer of 15 neurons, followed by softmax activation.
  - The classification output is based off cross-entropy loss.
- Resizing of the input images to 64x64, either by anisotropic rescaling or any other reasonably appropriate method.
- Splitting of the training set into training and validation set, with a 85/15 ratio.
- Use of the stochastic gradient descent with momentum optimization algorithm, minibatches of size 32, and starting weight values sampled from a normal distribution with mean 0 and standard deviation 0.01.

The network should reach a classification accuracy of ~30% on the test set. Discuss the choice of the learning rate and the stopping criterion used.

### Task 2
Task 2 consists in improving the results obtained in Task 1 by using data augmentation techniques, such as, but not limited to:
- Data augmentation using left-to-right reflections (should increase the accuracy to ~40%).
- Batch normalization before the ReLU layers.
- Changing the size of the convolution filters (3x3, 5x5, 7x7).
- Changing the initializations of the weights or the hyperparameters of the optimization algorithm, or employing the Adam optimization algorithm.
- Using dropout.
- Using an ensemble of networks (five to ten), and then picking the average of the outputs of the networks as the final classification.

Comment on any significant change in the classification accuracy after each modification. 

The final accuracy should be ~60%.

### Task 3
Task 3 consists in using transfer learning, for instance AlexNet, in the following two manners:
- Freezing the weights of all the layers except the last fully connected layer, and fine-tuning the last fully connected layer to the new dataset, based on the previous results obtained.
- By employing the pre-trained network as a feature extractor, accessing the activations of an intermediate layer of the network, train a multi-class linear SVM classifier on top of these features.

This should grant results of up to ~85% accuracy on the test set.

### Tasks 4, 5, 6, 7 (optional)
- **Task 4**: In tasks 2 and 3, an improvement in data augmentation, adding a random cropping of the images, and a small rotation and scaling of the images.
- **Task 5**: In task 2, adding more convolutional or fully connected layers to the network can improve the accuracy.
- **Task 6**: In task 3, employ a non-linear SVM classifier instead of a linear one.
- **Task 7**: In task 3, implement the multiclass SVM using the Error Correcting Output Code approach. [[Dietterich and Bakiri, 1994, James and Hastie, 1998]](#dietterich-and-bakiri-1994)

## References
- <a id="dietterich-and-bakiri-1994"></a>[Dietterich and Bakiri, 1994] Dietterich, T. G. and Bakiri, G. (1994). Solving multiclass learning problems via error-correcting output codes. Journal of artificial intelligence research, 2:263–286.