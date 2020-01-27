#!/usr/bin/python
import os
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import tensorflow
from tensorflow import keras
#Hands-on machine learning chapter 15. let's make a RNN that's really just a single neuron.
def make_waves(batch_size, steps):
    t = np.linspace(0, 1, steps)
    off1, off2 = np.random.rand(2, batch_size, 1)
    freq1, freq2 = 0.5, 0.2
    series = 0.5 * np.sin( (t+off1) * (freq1*10+10) )
    series += 0.5 * np.sin( (t+off2) * (freq2*30+30) )
    return series[..., np.newaxis].astype(np.float32)

def make_waveline(batch_size, steps):
    t = np.linspace(0, 1, steps)
    off1, off2 = np.random.rand(2, batch_size, 1)
    freq1, freq2 = 0.7, 0.2
    series = 0.5 * np.sin( (t+off1) * (freq2*10+10) )
    series += 0.5 * np.sin( (t+off2) * (freq1*30+30) )
    series += t*5
    return series[..., np.newaxis].astype(np.float32)

def simple_connected_network():
    model = keras.models.Sequential([\
            keras.layers.Flatten(input_shape=[100,1]),
            keras.layers.Dense(1)
        ])
    return model

def single_neuron_recurrent():
    model = keras.models.Sequential([  keras.layers.SimpleRNN(1, input_shape=[None,1])  ])
    #A RNN can process any number of previous time steps so can set input_shape to be None!
    return model

def multi_neuron_recurrent():
    model = keras.models.Sequential([\
            keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None,1]),
            keras.layers.SimpleRNN(20, return_sequences=True ),
            keras.layers.SimpleRNN(1)
            ])
    #A RNN can process any number of previous time steps so can set input_shape to be None!
    return model

if __name__ == "__main__":
    
    steps = 100
    modi = 10
    instances = 1000 * modi
    series = make_waves(instances, steps+1)
    #series = make_waveline(instances, steps+1)
   
    X_train, y_train = series[:700* modi, :steps], series[:700* modi, -1]
    X_valid, y_valid = series[700* modi:900* modi, :steps], series[700* modi:900* modi, -1]
    X_test, y_test = series[900* modi:1000* modi, :steps], series[900* modi:1000* modi, -1]
    
    #model = simple_connected_network()
    #model = single_neuron_recurrent()
    model = multi_neuron_recurrent()
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, 
            epochs=20,
            steps_per_epoch=50,
            validation_data=(X_valid, y_valid)
            )

    for i in range(1,30):
        plt.clf()
        plt.plot(X_test[i].flatten(), '.-')
        plt.plot(steps, y_test[i].flatten(), 'rx')
        plt.plot(steps, model.predict(X_test)[i], 'go')
        plt.legend(['data','actual','pred'])
        plt.savefig('image'+str(i)+'.png')


    
