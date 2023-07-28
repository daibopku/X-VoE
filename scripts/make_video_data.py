import time
import datetime
from absl import app
import os

import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging

if logging is None:
    # The logging module may have been unloaded when __del__ is called.
    log_fn = print
else:
    log_fn = logging.warning


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):  # if value ist tensor
        value = value.numpy()  # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_array(array):
    array = tf.io.serialize_tensor(array)
    return array


def parse_single_image(image, mask):

    #define the dictionary -- the structure -- of our single example
    data = {
        'raw_image': _bytes_feature(serialize_array(image)),
        'mask': _bytes_feature(serialize_array(mask))
    }
    #create an Example, wrapping the single features
    out = tf.train.Example(features=tf.train.Features(feature=data))

    return out


def parse_tfr_element(element):
    # use the same structure as above; it's kinda an outline of the structure we now want to create
    data = {
        'raw_image': tf.io.FixedLenFeature([], tf.string),
        'mask': tf.io.FixedLenFeature([], tf.string),
        'slot': tf.io.VarLenFeature(tf.float32),
    }
    content = tf.io.parse_single_example(element, data)
    raw_image = content['raw_image']
    raw_mask = content['mask']
    image_ori = tf.io.parse_tensor(raw_image, out_type=tf.uint8)
    # get our 'feature'-- our image -- and reshape it appropriately
    raw_mask = tf.io.parse_tensor(raw_mask, out_type=tf.uint8)
    return {"image_ori": image_ori, "raw_mask": raw_mask}


def write_images_to_tfr_short(data, writer):
    # filename= filename+".tfrecords"
    # writer = tf.io.TFRecordWriter(filename) #create a writer that'll store our data to disk
    # count = 0
    image = data['image_ori']
    mask = data['raw_mask']
    (B, N, _, _, _) = image.shape

    #get the data we want to write
    for i in range(B):
        for j in range(N):
            current_image = image[i, j]
            current_mask = mask[i, j]
            out = parse_single_image(image=current_image, mask=current_mask)
            writer.write(out.SerializeToString())
    writer.close()
    return 0


def main(argv):
    del argv
    videofilepath = "X-VoE/train/"
    imagefilepath = "X-VoE/image/"
    os.mkdir(imagefilepath)
    num_file = 100
    filename = [
        videofilepath + "train-part-{:0>3}.tfrecord".format(i)
        for i in range(num_file)
    ]
    ITEMS_PER_FILE = 1000
    dataset = tf.data.TFRecordDataset(filename, compression_type="GZIP")
    dataset = dataset.map(parse_tfr_element,
                          num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(ITEMS_PER_FILE, drop_remainder=True)

    def write_generator():
        i = 0
        iterator = iter(dataset)
        optional = iterator.get_next_as_optional()
        options = tf.io.TFRecordOptions(compression_type="GZIP")
        while optional.has_value().numpy():
            ds = optional.get_value()
            optional = iterator.get_next_as_optional()
            writer = tf.io.TFRecordWriter(
                imagefilepath + "/train-part-{:0>3}.tfrecord".format(i),
                options=options)  #compression_type='GZIP'
            i += 1
            yield ds, writer, i
        return

    start_time = time.time()
    for data, wri, i in write_generator():
        write_images_to_tfr_short(data, wri)
        log_fn("Num: {}, Time: {}".format(
            i, datetime.timedelta(seconds=time.time() - start_time)))


if __name__ == "__main__":
    app.run(main)