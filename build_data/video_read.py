import os
import tensorflow as tf


# @tf.function
def build_data(split, shuffle=False):
    file_path_base = 'X-VoE/'

    # file_path_base = '/scratch/LargeTaskPlatform/daibo/VoE_dataset/'
    def _parse_tfr_element(element):
        # use the same structure as above; it's kinda an outline of the structure we now want to create
        data = {
            'raw_image': tf.io.FixedLenFeature([], tf.string),
            'mask': tf.io.FixedLenFeature([], tf.string),
            'slot': tf.io.VarLenFeature(tf.float32),
        }

        content = tf.io.parse_single_example(element, data)

        raw_image = content['raw_image']
        raw_mask = content['mask']
        raw_slot = content['slot']

        image_ori = tf.io.parse_tensor(raw_image, out_type=tf.uint8)
        image = tf.cast(image_ori, tf.float32)
        image = ((image / 255.0) - 0.5) * 2.0

        # get our 'feature'-- our image -- and reshape it appropriately
        raw_mask = tf.io.parse_tensor(raw_mask, out_type=tf.uint8)
        mask = tf.cast(raw_mask, tf.float32)
        mask = tf.clip_by_value(mask, 0.0, 1.0)
        slot = raw_slot.values
        return {
            "image_ori": image_ori,
            "raw_mask": raw_mask,
            "image": image,
            "mask": mask,
            "slot": slot
        }

    AUTOTUNE = tf.data.AUTOTUNE
    if split == "train":
        num_file = 100
        file_path = os.path.join(file_path_base, 'train')
        filename = [
            os.path.join(file_path, "train-part-{:0>3}.tfrecord".format(i))
            for i in range(num_file)
        ]
    elif split in ["collision", "blocking", "continuity"]:
        num_file = 6
        file_path = os.path.join(file_path_base, "test")
        file_path = os.path.join(file_path, split)
        filename = [
            os.path.join(file_path, "eval-part-{:0>3}.tfrecord".format(i))
            for i in range(num_file)
        ]
    elif split in ["permanence"]:
        num_file = 4
        file_path = os.path.join(file_path_base, "test")
        file_path = os.path.join(file_path, split)
        filename = [
            os.path.join(file_path, "eval-part-{:0>3}.tfrecord".format(i))
            for i in range(4)
        ]
    elif split == "eval":
        num_file = 4
        file_path = os.path.join(file_path_base, "test")
        eval_list = ["collision", "blocking", "permanence", "continuity"]
        filename = [
            file_path + i + '/' + "eval-part-000.tfrecord" for i in eval_list
        ]
    else:
        raise ValueError("Error dataset type")
    if shuffle:
        filename = tf.data.Dataset.from_tensor_slices(filename)
        filename = filename.shuffle(num_file)
        ds = filename.interleave(
            lambda x: tf.data.TFRecordDataset(x, compression_type="GZIP"),
            cycle_length=num_file,
            block_length=1)
        ds = ds.shuffle(1000)
    else:
        ds = tf.data.TFRecordDataset(filename, compression_type="GZIP")
    ds = ds.map(_parse_tfr_element, num_parallel_calls=AUTOTUNE)
    return ds


def load_data(batch_size, split, **kwargs):
    ds = build_data(split=split, **kwargs)
    ds = ds.batch(batch_size, drop_remainder=True)
    return ds
