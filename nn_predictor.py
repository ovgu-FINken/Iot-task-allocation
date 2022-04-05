import numpy as np
from keras import layers
from tensorflow import keras

class nnPredictor:
    def __init__(self):
        self.model = keras.models.load_model('model')
    def Predict(self, sequence):
        temp = []
        temp = np.zeros((20, 2))
        for i, val in enumerate(sequence):
            temp[-1-i][0] = val[0]
            temp[-1-i][1] = val[1]
            if i==19:
                break
        temp = np.asarray([temp])
        predictNextNumber = self.model.predict(temp, verbose=0)
        return (predictNextNumber[0][0],predictNextNumber[0][1])
