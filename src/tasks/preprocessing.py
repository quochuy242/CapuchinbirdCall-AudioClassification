from audio_to_spectrogram import get_spectrogram
import tensorflow as tf
import tensorflow_io as tfio


def load_wav_16k_mono(filename):
    """
    Hàm dùng để load các file có đuôi là .wav và
    biến chúng từ tần số 44100 kHz thành 16000 kHz,
    đồng thời cài đặt chúng ở chế độ âm thanh đơn hướng (mono)
    """
    file_contents = tf.io.read_file(filename)

    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)

    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav


def preprocess(filepath, label):
    wav = load_wav_16k_mono(filepath)
    wav = wav[:48000]
    zero_padding = tf.zeros([48000] - tf.shape(wav), dtype=tf.float32)
    wav = tf.concat([zero_padding, wav], axis=0)

    spectrogram = get_spectrogram(wav, frame_length=320, frame_step=32)
    return spectrogram, label
