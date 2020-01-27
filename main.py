#!/usr/bin/python
import os
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt


#stored in $HOME/<user>/.keras/datasets/jena_climate_2009_2016.csv
def dl_weather_data():
    zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True)
    csv_path, _ = os.path.splitext(zip_path)
    return(csv_path)


def univariate_data(dataset, start_index, end_index, history_size, target_size):
    data = []
    labels = []
    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i-history_size, i)
        # Reshape data from (history_size,) to (history_size, 1)
        data.append(np.reshape(dataset[indices], (history_size, 1)))
        labels.append(dataset[i+target_size])
    return np.array(data), np.array(labels)

def create_time_steps(length):
    return list(range(-length,0))

def printPlot(plot_data, delta, figname):
    labels = ['hist', 'Truth', 'pred']
    marker = ['.-','rx','go']
    time_steps = create_time_steps(plot_data[0].shape[0])
    if delta:
        future = delta
    else:
        future = 0
    plt.clf()
    plt.title('my plot')
    for i, x in enumerate(plot_data):
        if (i>0):#plot the future and pred
            plt.plot(future, plot_data[i].flatten(), marker[i], markersize=10, label = labels[i])
        else:#print past data
            plt.plot(time_steps, plot_data[i].flatten(), marker[i], label = labels[i])
    plt.xlabel('Time step')
    plt.xlim([time_steps[0], (future+5)*2])
    plt.legend(labels)
    plt.savefig(figname)
    



#def uniVariative(var):




if __name__ == "__main__":
    csv_path = dl_weather_data()
    data = pd.read_csv(csv_path)
    uni_data = data["T (degC)"]
    uni_data.index = data["Date Time"]
    #Want data to be an array for input.
    uni_data=uni_data.values
    rows_use=200000
    #standardation: is subtracting the mean and dividing by the standard deviation
    uni_train_mean = uni_data[:rows_use].mean()
    uni_train_stdDev = uni_data[:rows_use].std()
    uni_data = (uni_data-uni_train_mean)/uni_train_stdDev
    #model given 20 last items to learn from
    uni_past_history = 20
    uni_future_target = 0
    # cut from start to item.
    x_train_uni, y_train_uni = univariate_data(uni_data, 0, rows_use, uni_past_history, uni_future_target)
    # cut from item till end
    x_val_uni, y_val_uni = univariate_data(uni_data, rows_use, None, uni_past_history, uni_future_target)

    print("x_train_uni, y_train_uni", x_train_uni.shape, y_train_uni.shape)
    print("x_val_uni, y_val_uni", x_val_uni.shape, y_val_uni.shape)

    printPlot([x_train_uni[0], y_train_uni[0]],0, 'noPred.png')
    printPlot([x_train_uni[0], y_train_uni[0], np.mean(x_train_uni[0])],0 , 'plotMean.png')
    
    
    #OK now doing RNN. this is the "Long Short Term Memory" RNN layer.
    
    BATCH_SIZE = 256
    BUFFER_SIZE = 10000
    
    #what does this tuff do?
    train_uni = tf.data.Dataset.from_tensor_slices( ( x_train_uni, y_train_uni) )

    train_uni = train_uni.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

    val_uni = tf.data.Dataset.from_tensor_slices( (x_val_uni, y_val_uni) )

    val_uni = val_uni.batch(BATCH_SIZE).repeat()

    simple_lstm = tf.keras.models.Sequential([ tf.keras.layers.LSTM(8, input_shape = x_train_uni.shape[-2:]),\
            tf.keras.layers.Dense(1)])

    simple_lstm.compile(optimizer='adam', loss='mse')


    #check the model output...
    EVALUATION_INTERVAL = 200

    simple_lstm.fit(\
        train_uni,
        epochs=10,
        steps_per_epoch=EVALUATION_INTERVAL,
        validation_data = val_uni,
        validation_steps=50
            )

    figs=[]
    for i in range(0,10):
        figs.insert(i, str(i)+".png")

    #take 10 predictions
    i=0
    for x, y in val_uni.take(10):
        pltData = [x[0].numpy(), y[0].numpy(), simple_lstm.predict(x)[0]]
        printPlot(pltData, 0, figs[i])
        i+=1
    


    








