# Neural Network implemented with MNIST Dataset

This project implements a neural network using the Numpy library to classify handwritten digits from the MNIST dataset. The neural network architecture consists of three layers with different activation functions: tanh, ReLU, and softmax.

The MNIST dataset contains images of handwritten digits from 0 to 9, and our goal is to build a neural network that can accurately classify these digits.

The first layer of the neural network is the input layer, which takes the pixel values of the input image as input. These pixel values are flattened into a vector and fed into the next layer.

The second layer is the hidden layer, which applies the tanh activation function to the input. The tanh activation function squashes the input values between -1 and 1, allowing the network to model non-linear relationships in the data.

The third layer is also a hidden layer, but it uses the ReLU (Rectified Linear Unit) activation function. The ReLU activation function sets all negative input values to zero, while keeping the positive values unchanged. This helps the network learn sparse representations and can speed up training.

Finally, the last layer is the output layer, which applies the softmax activation function. The softmax function takes the inputs and normalizes them into a probability distribution over the possible classes. This allows us to interpret the outputs as the probabilities of the input image belonging to each digit class.

During training, we use a technique called backpropagation to adjust the weights and biases of the network based on the errors between the predicted outputs and the true labels. We update the parameters using gradient descent, where the gradients are computed using the chain rule.

To evaluate the performance of the network, we calculate the cross-entropy loss between the predicted probabilities and the true labels. Lower loss indicates better performance. We use this loss to update the network's parameters and iteratively train the network until convergence.

By implementing this neural network from scratch using only Numpy, we have gained a deeper understanding of the underlying principles and mathematics behind neural networks. Although other ML frameworks like PyTorch, TensorFlow, and Keras provide convenient abstractions for neural network development, implementing the network using Numpy allows us to have full control over the model and better grasp the inner workings of neural networks.
