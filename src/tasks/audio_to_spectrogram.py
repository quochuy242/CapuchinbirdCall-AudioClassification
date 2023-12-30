import os
import tensorflow as tf
import tensorflow_io as tfio


def get_spectrogram(wav, frame_length, frame_step):
    spectrogram = tf.signal.stft(wav, frame_length=frame_length, frame_step=frame_step)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram
