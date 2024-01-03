import os
import sys
import tensorflow as tf

from tensorflow import keras
from keras.models import Sequential
from keras.layers import (
    Dense,
    Dropout,
    MaxPooling2D,
    Conv2D,
    Flatten,
    Input,
    Resizing,
    Normalization,
)
from keras.metrics import Accuracy, Recall, Precision
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

sys.path.insert(0, "D:/Python Project/DeepAudioClassification/src")
from tasks.data_pipeline import data_pipeline, create_data
from tasks.split_data import split_data

Dataset = tf.data.Dataset
AUTOTUNE = tf.data.AUTOTUNE


class Audio_Dataset:
    def __init__(self, pos_path, neg_path, train_rate, val_rate) -> None:
        self.pos_path = pos_path
        self.neg_path = neg_path
        self.train_rate = train_rate
        self.val_rate = val_rate
        self.test_rate = 1 - train_rate - val_rate

    def pipeline(self):
        data = create_data(self.pos_path, self.neg_path)
        data = data_pipeline(data, batch_size=64)
        train_data, validation_data, test_data = split_data(
            data, self.train_rate, self.val_rate, self.test_rate
        )
        return train_data, validation_data, test_data


class CNN_Model:
    def __init__(self, input_shape, resize_height, resize_width) -> None:
        self.model = Sequential()
        self.model.add(Input(shape=input_shape))
        self.model.add(Resizing(resize_height, resize_width))
        self.model.add(Normalization())

    def add_conv2d(self, filters, kernel_size, activation):
        self.model.add(
            Conv2D(filters=filters, kernel_size=kernel_size, activation=activation)
        )

    def add_maxpooling2d(self, pool_size):
        self.model.add(MaxPooling2D(pool_size=pool_size))

    def add_flatten(self):
        self.model.add(Flatten())

    def add_dense(self, units, activation):
        self.model.add(Dense(units=units, activation=activation))

    def add_dropout(self, rate):
        self.model.add(Dropout(rate))

    def model_summary(self):
        return self.model.summary()

    def model_compile(self, optimizer, loss, metrics):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def model_fit(self, train_data, validation_data, epochs, callbacks: list):
        history = self.model.fit(
            train_data,
            validation_data=validation_data,
            epochs=epochs,
            callbacks=callbacks,
        )
        return history

    def model_evaluate(self, test_data):
        return self.model.evaluate(test_data)

    def model_save(self, path):
        self.model.save(path)


if __name__ == "__main__":
    sys.path.insert(0, "D:/Python Project/DeepAudioClassification")
    PARSED_DIR = os.path.join("data", "Parsed_Capuchinbird_Clips")
    NOT_PARSED_DIR = os.path.join("data", "Parsed_Not_Capuchinbird_Clips")
    data = Audio_Dataset(
        pos_path=PARSED_DIR, neg_path=NOT_PARSED_DIR, train_rate=0.9, val_rate=0.1
    )
    train_ds, val_ds, test_ds = data.pipeline()

    model = CNN_Model(
        input_shape=(1491, 257, 1),
        resize_height=32,
        resize_width=32,
    )
    model.add_conv2d(filters=32, kernel_size=(3, 3), activation="relu")
    model.add_maxpooling2d(pool_size=(2, 2))
    # model.add_dropout(rate=0.2)

    model.add_conv2d(filters=16, kernel_size=(3, 3), activation="relu")
    # model.add_maxpooling2d(pool_size=(2, 2))
    # model.add_dropout(rate=0.2)

    model.add_flatten()
    model.add_dense(units=128, activation="relu")
    model.add_dense(units=1, activation="sigmoid")

    model.model_compile(
        optimizer=Adam(learning_rate=0.00001),
        loss="binary_crossentropy",
        metrics=[Accuracy(), Recall(), Precision()],
    )
    model.model_summary()
    early_stopping = EarlyStopping(
        monitor="val_recall",
        patience=200,
        verbose=1,
        mode="max",
        restore_best_weights=True,
    )
    model_checkpoint = ModelCheckpoint(
        filepath="weights/best_model.h5",
        monitor="val_recall",
        save_best_only=True,
        mode="max",
    )
    model.model_fit(
        train_data=train_ds,
        validation_data=val_ds,
        epochs=2000,
        callbacks=[early_stopping, model_checkpoint],
    )
