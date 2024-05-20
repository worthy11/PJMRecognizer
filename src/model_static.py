from keras.layers import Dense, Flatten
from keras.models import Sequential, load_model
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
import tensorflow as tf
import tf2onnx
import onnx
import os
from data_loader import *

class Model():
    def __init__(self, classes: dict, epochs: int=10, batch_size: int=1, learning_rate: float=0.00001, from_checkpoint: bool=False):
        self.classes = classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = ReduceLROnPlateau(monitor='val_accuracy',
                                               patience=2,
                                               verbose=1,
                                               factor=0.5,
                                               min_lr=learning_rate)
        if os.path.getsize('./src/weights.keras') > 0 and from_checkpoint:
            self.model = load_model('src/weights.keras')
            print('Loaded weights from checkpoint')
        else:
            self.model = self.InitializeFromArchitecture()
            print('Initialized new model')
        self.model.compile(loss='categorical_crossentropy', optimizer='adam')

    def InitializeFromArchitecture(self):
        model = Sequential()
        model.add(Flatten())
        model.add(Dense(units=len(self.classes), activation='softmax'))
        return model
    
    def TrainModel(self, to_checkpoint: bool=True):
        (train_set, train_labels), (test_set, test_labels) = LoadData(len(self.classes))

        if to_checkpoint:
            filepath = './src/weights.keras'
            checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
            callbacks_list = [checkpoint]
            history = self.model.fit(train_set,
                                     train_labels,
                                     batch_size=self.batch_size,
                                     epochs=self.epochs,
                                     validation_data=(test_set, test_labels),
                                     callbacks = callbacks_list)
            new_model = load_model(filepath)
            self.model = new_model
            self.model.compile(loss='categorical_crossentropy', optimizer='adam')

            self.model.output_names=['output']
            input_signature = [tf.TensorSpec(self.model.inputs[0].shape, self.model.inputs[0].dtype, name='x')]
            onnx_model, _ = tf2onnx.convert.from_keras(self.model, input_signature, opset=13)

            onnx.save(onnx_model, "pjmrecognizer.onnx")

        else:
            history = self.model.fit(train_set,
                                     train_labels,
                                     batch_size=self.batch_size,
                                     epochs=self.epochs,
                                     validation_data=(test_set, test_labels),
                                     callbacks = [self.learning_rate])
        results = self.model.evaluate(test_set, test_labels)
        return (history, results)
        
    def Predict(self, sample: np.array) -> str:
        predictions = self.model(sample)
        label = self.classes[np.argmax(predictions)]
        confidence = np.max(predictions) / np.sum(predictions)
        return label, confidence 