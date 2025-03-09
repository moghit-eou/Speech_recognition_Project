import numpy as np
from random import random


class MLP(object):
    
    def __init__(self, num_inputs=3, hidden_layers=[3, 5], num_outputs=2):
        
        self.num_inputs = num_inputs
        self.hidden_layers = hidden_layers
        self.num_outputs = num_outputs

        # create a generic representation of the layers
        layers = [num_inputs] + hidden_layers + [num_outputs]

        # create random connection weights for the layers
        weights = []
        for i in range(len(layers) - 1):
            w = np.random.rand(layers[i], layers[i + 1])
            weights.append(w)
        self.weights = weights
        
        # save derivatives per layer
        derivatives = []
        for i in range(len(layers) - 1):
            d = np.zeros((layers[i], layers[i + 1]))
            derivatives.append(d)
        self.derivatives = derivatives

        # save activations per layer
        activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)
        self.activations = activations


    def forward_propagate(self, inputs):
        # the input layer activation is just the input itself
        activations = inputs

        # save the activations for backpropogation
        self.activations[0] = activations

        # iterate through the network layers
        for i, w in enumerate(self.weights):
            # calculate matrix multiplication between previous activation and weight matrix
            net_inputs = np.dot(activations, w)

            # apply sigmoid activation function
            activations = self._sigmoid(net_inputs)

            # save the activations for backpropogation
            self.activations[i + 1] = activations

        # return output layer activation
        return activations
        
    def back_propagate(self , error , verbose = False):
        
        # dE/dW_i = (y - a_[i+1]) * s'(h_[i+1]) * a_i
        # s'(h_[i+1]) = s(h_[i+1]) * (1 - s(h_[i+1]))
        # s(h_[i+1]) = a_[i+1]

        # error = ( y - a[i + 1] )
        # delta = error * sigmoid_derivative

        # dE/dW_[i-1] = (y - a_[i+1]) * s'(h_[i+1]) * W_i * s'(h_i) * a_[i-1]
        #error =  (y - a_[i+1]) * s'(h_[i+1]) * W_i
        
        for i in reversed( range(len(self.derivatives))):   # this is like starting from the last index
            activations = self.activations[i + 1]
                
            delta = error * self._sigmoid_derivative( activations ) 
            current_activations = self.activations[i] # a[i]

            #this is for simplifying the multiplication
            delta_reshaped = delta.reshape( delta.shape[0] , -1 ) # [0 , 1 , 2] --> [ [0] , [1] , [2] ] as matrix shape
            current_activations_reshaped = current_activations.reshape( current_activations.shape[0] , -1 )

            self.derivatives[i] = np.dot(current_activations_reshaped , delta_reshaped.T ) 

            if verbose:
                print("Derivative for W{} = {}".format(i , self.derivatives[i] ))
            error = np.dot ( delta , self.weights[i].T )
        return error 


    def train(self, inputs, targets, epochs, learning_rate):
        """Trains model running forward prop and backprop
        Args:
            inputs (ndarray): X
            targets (ndarray): Y
            epochs (int): Num. epochs we want to train the network for
            learning_rate (float): Step to apply to gradient descent
        """
        # now enter the training loop
        for i in range(epochs):
            sum_errors = 0

            # iterate through all the training data
            for j, input in enumerate(inputs):
                target = targets[j]

                # activate the network!
                output = self.forward_propagate(input)

                error = target - output

                self.back_propagate(error)

                # now perform gradient descent on the derivatives
                # (this will update the weights
                self.gradient_descent(learning_rate)

                # keep track of the MSE for reporting later
                sum_errors += self._mse(target, output)

            # Epoch complete, report the training error
            print("Error: {} at epoch {}".format(sum_errors / len(items), i+1))

        print("Training complete!")
        print("=====")

        
    def gradient_descent(self, learningRate=1):
        """Learns by descending the gradient
        Args:
            learningRate (float): How fast to learn.
        """
        # update the weights by stepping down the gradient
        for i in range(len(self.weights)):
            weights = self.weights[i]
            derivatives = self.derivatives[i]
            weights += derivatives * learningRate




    def _sigmoid(self, x):
        """Sigmoid activation function
        Args:
            x (float): Value to be processed
        Returns:
            y (float): Output
        """

        y = 1.0 / (1 + np.exp(-x))
        return y


    def _sigmoid_derivative(self, x):
        """Sigmoid derivative function
        Args:
            x (float): Value to be processed
        Returns:
            y (float): Output
        """
        return x * (1.0 - x)


    def _mse(self, target, output):
        """Mean Squared Error loss function
        Args:
            target (ndarray): The ground trut
            output (ndarray): The predicted values
        Returns:
            (float): Output
        """
        return np.average((target - output) ** 2)

if __name__ == "__main__":

    # create a dataset to train a network for the sum operation
    items = np.array([[random()/2 for _ in range(2)] for _ in range(1000)])
    targets = np.array([[i[0] + i[1]] for i in items])
    # create a Multilayer Perceptron with one hidden layer
    mlp = MLP(2, [10], 1)

    # train network
    mlp.train(items, targets, 52, 0.1)

    # create dummy data
    input = np.array([0.3, 0.1])   
    target = np.array([0.4])

    # get a prediction
    output = mlp.forward_propagate(input)

    print()
    print("Our network believes that {} + {} is equal to {}".format(input[0], input[1], output[0]))

    