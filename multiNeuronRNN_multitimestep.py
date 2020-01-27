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


def multi_neuron_recurrent():
    model = keras.models.Sequential([\
            keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None,1]),
            keras.layers.SimpleRNN(20),
            keras.layers.Dense(10)
            ])
    return model

if __name__ == "__main__":
    
    steps = 100
    m = 20 # scaling
    instances = 1000 * m
    
    series = make_waves(instances, steps + 10)
    #series = make_waveline(instances, steps+10)
    print('series size: ', series.shape)

    X_train, y_train = series[:700*m, :steps], series[:700*m, -10:]
    X_valid, y_valid = series[700*m:900*m, :steps], series[700*m:900*m, -10:]
    X_test, y_test = series[900*m:1000*m, :steps], series[900*m:1000*m, -10:]
    #y_train = y_train[...,np.newaxis]
    #y_valid = y_valid[...,np.newaxis]
    #y_test = y_test[...,np.newaxis]
    print(X_train.shape)
    print(y_train.shape)


    #correct from here down
    model = multi_neuron_recurrent()
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, 
            epochs=20, steps_per_epoch=20,
            validation_data=(X_valid, y_valid)
            )

    for i in range(1,10):
        plt.clf()
        plt.plot(X_test[i].flatten(), '.-')
        plt.plot(range(steps, 10 + steps), y_test[i].flatten(), 'rx')
        plt.plot(range(steps, 10 + steps), model.predict(X_test)[i], 'go')
        plt.legend(['data','actual','pred'])
        plt.savefig('image'+str(i)+'.png')


    
