import sys
import tensorflow as tf

sys.path.insert(0, "D:/Python Project/DeepAudioClassification/src/tasks")
from preprocessing import preprocess


def create_data(positive_dir, negative_dir):
    positive_ds = tf.data.Dataset.list_files(positive_dir)
    negative_ds = tf.data.Dataset.list_files(negative_dir)

    positive_labels = tf.data.Dataset.from_tensor_slices(tf.ones(len(positive_ds)))
    negative_labels = tf.data.Dataset.from_tensor_slices(tf.zeros(len(negative_ds)))

    positive_data = tf.data.Dataset.zip((positive_ds, positive_labels))
    negative_data = tf.data.Dataset.zip((negative_ds, negative_labels))

    data = tf.data.Dataset.concatenate(positive_data, negative_data)
    return data


def data_pipeline(data, batch_size: int):
    data = data.map(preprocess)
    data = data.cache()
    data = data.shuffle(buffer_size=1000)
    data = data.batch(batch_size)
    data = data.prefetch(buffer_size=tf.data.AUTOTUNE)

    return data
