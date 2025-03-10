from sklearn.model_selection import train_test_split
import numpy as np
from random import random
import tensorflow as tf


def generate_dataset(num_samples, test_size=0.33):
 

    # build inputs/targets for sum operation: y[0][0] = x[0][0] + x[0][1]
    x = np.array([[random()/2 for _ in range(2)] for _ in range(num_samples)])
    y = np.array([[i[0] + i[1]] for i in x])

    # split dataset into test and training sets

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    return x_train , x_test, y_train , y_test

if __name__ == "__main__":
    x_train , x_test , y_train , y_test = generate_dataset(4000 , 0.3)
    
    #building the model : 2 | 5 | 1
    model = tf.keras.Sequential([
    tf.keras.layers.Dense(5, input_dim=2, activation="sigmoid"),
    tf.keras.layers.Dense(1, activation="sigmoid")
    ])


    # compile 
    optimiser = tf.keras.optimizers.SGD(learning_rate = 0.1) # stochastique gradient descent
    model.compile(optimizer = optimiser , loss = "MSE") # mean square error

    #train model
    model.fit(x_train , y_train , epochs = 100)
    #evaluate
    print("\n\n\n-----------------------------------------------\n\n")
    model.evaluate(x_test , y_test , verbose = True)
    data = np.array([ 
        [0.1 , 0.2] , 
        [0.2 , 0.2]
    ])

    print("\n\n-------preidction---------\n\n ")
    predictions = model.predict(data)
    
    for d , p in zip(data , predictions):
        print("{} + {} = {}".format(d[0] , d[1] , p[0] ) )
