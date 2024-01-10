"""Data utils."""
# from scipy import rand
import tensorflow as tf
import os

# import tensorflow_datasets as tfds
# from tensorflow.python.ops import gen_io_ops

# tf.config.run_functions_eagerly(True)


# @tf.function
def build_data(train_size, test_size, shuffle=False):
    num_file = 100

    def _parse_tfr_element(element):
        # use the same structure as above; it's kinda an outline of the structure we now want to create
        data = {
            'raw_image': tf.io.FixedLenFeature([], tf.string),
            'mask': tf.io.FixedLenFeature([], tf.string),
        }

        content = tf.io.parse_single_example(element, data)

        raw_image = content['raw_image']
        raw_mask = content['mask']

        # get our 'feature'-- our image -- and reshape it appropriately
        image = tf.io.parse_tensor(raw_image, out_type=tf.uint8)
        image = tf.cast(image, tf.float32)
        image = ((image / 255.0) - 0.5) * 2.0
        mask = tf.io.parse_tensor(raw_mask, out_type=tf.uint8)
        mask = tf.cast(mask, tf.float32)
        image = tf.expand_dims(image, axis=0)
        mask = tf.expand_dims(mask, axis=0)
        return {"image": image, "mask": mask}

    AUTOTUNE = tf.data.AUTOTUNE
    file_path = 'X-VoE/image/'
    filename = [
        os.path.join(file_path, "train-part-{:0>3}.tfrecord".format(i))
        for i in range(num_file)
    ]
    # filename = os.listdir(file_path)
    filename = tf.data.Dataset.from_tensor_slices(filename)
    filename = filename.shuffle(num_file)
    ds = filename.interleave(
        lambda x: tf.data.TFRecordDataset(x, compression_type="GZIP"),
        cycle_length=num_file,
        block_length=1)
    train_ds = ds.take(train_size)
    test_ds = ds.skip(train_size)
    test_ds = ds.take(test_size)
    if shuffle:
        train_ds = train_ds.shuffle(10000)
        test_ds = test_ds.shuffle(10000)
    train_ds = train_ds.map(_parse_tfr_element, num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.map(_parse_tfr_element, num_parallel_calls=AUTOTUNE)
    return train_ds, test_ds


def load_data(batch_size, train_size=60000, test_size=20000, **kwargs):
    train_ds, test_ds = build_data(train_size=train_size,
                                   test_size=test_size,
                                   **kwargs)
    train_ds = train_ds.batch(batch_size, drop_remainder=True)
    test_ds = test_ds.batch(batch_size, drop_remainder=True)
    return train_ds, test_ds
