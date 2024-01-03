import tensorflow as tf


def split_data(
    data,
    train_rate: float,
    validation_rate: float,
    test_rate: float = 0,
):
    len_data = len(data)
    train_data = data.take(int(len_data * train_rate))
    if test_rate != 0:
        test_data = data.skip(int(len_data * train_rate)).take(
            int(len_data * test_rate)
        )
        validation_data = data.skip(int(len_data * (train_rate + test_rate))).take(
            int(len_data * (1 - train_rate - test_rate))
        )
    else:
        validation_data = data.skip(int(len_data * train_rate)).take(
            int(len_data * (1 - train_rate))
        )

    return train_data, validation_data, test_data
