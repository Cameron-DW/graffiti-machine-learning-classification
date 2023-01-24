# Python module to create a multilayer perceptron neural network classifier for the graffiti images
from tensorflow import keras
from tensorflow.keras.layers import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
usualCallback = EarlyStopping()
overfit_callback = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, mode='min')


# a NeuralNetwork class to take extracted image feature inputs and classify the images
class NeuralNetwork:
    def __init__(self, input_data, target_data):
        self.input_full = input_data.astype('float32')
        self.target_full = target_data.astype('float32')
        self.input_train_data,\
            self.input_test_data,\
            self.target_train_data,\
            self.target_test_data = train_test_split(input_data, target_data, test_size=0.15)
        self.input_num = input_data.shape[1]  # i think it is [1] and not [0]
        self.classifier = keras.Sequential()

    # Function to setup the neural network, defining its architecture
    def setup(self):
        print(self.input_num)
        self.classifier.add(Dense(10, input_dim=self.input_num, activation='relu'))
        self.classifier.add(Dense(10, activation='relu'))
        self.classifier.add(Dense(10, activation='relu'))
        self.classifier.add(Dense(10, activation='relu'))
        self.classifier.add(Dense(1, activation='sigmoid'))
        self.classifier.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])

    # Function to train the neural network
    def train(self, batch_size=10, epochs=1000):
        history = self.classifier.fit(self.input_train_data, self.target_train_data, batch_size, epochs, validation_split=0.15*0.85, callbacks=[overfit_callback])
        print(history.history.keys())

        confusion_m = self.get_confusion_matrix()
        print(confusion_m)
        print(self.evaluation_metrics(confusion_m))

        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('classifier accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.yticks((0.0,0.1,0.2,0.3,0.4,0.5, 0.6, 0.7, 0.8, 0.9, 1.0))
        plt.legend(['training data', 'validation data'], loc='lower right')
        plt.show()

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('classifier loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.yticks((0.0,0.1,0.2,0.3,0.4,0.5, 0.6, 0.7, 0.8, 0.9, 1.0))
        plt.legend(['training data', 'validation data'], loc='upper right')
        plt.show()

    # Function to get the confusion matrix for the trained neural network
    def get_confusion_matrix(self):
        target_predict = self.classifier.predict(self.input_test_data)
        target_predict = np.around(target_predict)
        confusion_m = confusion_matrix(self.target_test_data, target_predict)
        return confusion_m

    # Function to return the evaluation metrics (accuracy, precision, recall and F1 score) for the trained neural
    # network
    def evaluation_metrics(self, confusion_m):
        list_confusion = confusion_m.flatten()
        num_tests = sum(list_confusion)
        true_negative = confusion_m[0, 0]
        true_positive = confusion_m[1, 1]
        false_positive = confusion_m[0, 1]
        false_negative = confusion_m[1, 0]

        print(f"accuracy: {round((true_negative+true_positive)/num_tests,3)}")
        precision = true_positive/(true_positive+false_positive)
        print(f"precision:  {round(precision, 3)}")
        recall = true_positive/(true_positive+false_negative)
        print(f"recall:  {round(recall, 3)}")
        print(f"F1 Score: {round(2*((precision*recall)/(precision+recall)), 3)}")
