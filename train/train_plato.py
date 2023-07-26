import datetime
import time

from absl import app
from absl import flags
import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
from tensorflow_probability import distributions as tfd
from tensorboardX import SummaryWriter

import build_data.video_read as data_utils
import model.model as model_utils

FLAGS = flags.FLAGS
flags.DEFINE_string("model_dir", "checkpoint/PLATO/",
                    "Where to save the checkpoints.")
flags.DEFINE_string("tensorboard_dir", "tensorboard/PLATO/",
                    "Where to save the process.")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_integer("batch_size", 500, "Batch size for the model.")
flags.DEFINE_integer("batch_size_test", 100, "Batch size for the model.")
flags.DEFINE_integer("num_frames", 15, "Number of frames in video.")
flags.DEFINE_integer("num_slots", 8, "Number of slots in Slot Attention.")
flags.DEFINE_integer("slot_size", 32, "denth of object slot.")
flags.DEFINE_float("learning_rate", 0.0004, "Learning rate.")
flags.DEFINE_integer("max_epochs", 160, "Number of training epochs.")
flags.DEFINE_integer("max_steps", 40000, "Number of training steps.")
flags.DEFINE_integer("warmup_steps", 1000,
                     "Number of warmup steps for the learning rate.")
flags.DEFINE_float("decay_rate", 0.5, "Rate for the learning rate decay.")
flags.DEFINE_integer("decay_steps", 10000,
                     "Number of steps for the learning rate decay.")


class TensorboardViz(object):
    def __init__(self, logdir):
        self.logdir = logdir
        self.writter = SummaryWriter(self.logdir)

    def text(self, _text):
        # Enhance line break and convert to code blocks
        _text = _text.replace('\n', '  \n\t')
        self.writter.add_text('Info', _text)

    def update(self, mode, it, eval_dict):
        self.writter.add_scalars(mode, eval_dict, global_step=it)

    def flush(self):
        self.writter.flush()


if logging is None:
    # The logging module may have been unloaded when __del__ is called.
    log_fn = print
else:
    log_fn = logging.warning


def get_video_ED(num_frames, resolution, batch_size, num_slots, slot_size,
                 encode_type, decode_type, file):
    model_pre = model_utils.build_percept_model(resolution,
                                                batch_size * num_slots *
                                                num_frames,
                                                num_channels=4,
                                                slot_size=slot_size,
                                                decode_type=decode_type,
                                                encode_type=encode_type)
    ckpt_pre = tf.train.Checkpoint(network=model_pre)
    ckpt_manager_pre = tf.train.CheckpointManager(ckpt_pre,
                                                  directory=file,
                                                  max_to_keep=5)
    if ckpt_manager_pre.latest_checkpoint:
        ckpt_pre.restore(ckpt_manager_pre.latest_checkpoint).expect_partial()
        log_fn("Restored from {}".format(ckpt_manager_pre.latest_checkpoint))
    model_enc = model_pre.get_layer("ObjectEncoder")
    model_dec = model_pre.get_layer("ObjectDecoder")
    return model_enc, model_dec, model_pre


def main(argv):
    del argv
    # Hyperparameters of the model.
    batch_size = FLAGS.batch_size
    batch_size_test = FLAGS.batch_size_test
    num_frames = FLAGS.num_frames
    num_slots = FLAGS.num_slots
    slot_size = FLAGS.slot_size
    base_learning_rate = FLAGS.learning_rate
    max_epochs = FLAGS.max_epochs
    max_steps = FLAGS.max_steps
    warmup_steps = FLAGS.warmup_steps
    decay_rate = FLAGS.decay_rate
    decay_steps = FLAGS.decay_steps
    tf.random.set_seed(FLAGS.seed)

    resolution = (128, 128)

    decode_type = "SBTD"
    encode_type = "ViT"

    mirrored_strategy = tf.distribute.MirroredStrategy()
    log_fn('Number of devices: {}'.format(
        mirrored_strategy.num_replicas_in_sync))

    viz = TensorboardViz(logdir=FLAGS.tensorboard_dir)

    train_ds = data_utils.load_data(batch_size, split="train", shuffle=True)
    train_ds = mirrored_strategy.experimental_distribute_dataset(train_ds)

    test_ds = data_utils.load_data(batch_size_test,
                                   split="eval",
                                   shuffle=False)
    test_ds = mirrored_strategy.experimental_distribute_dataset(test_ds)

    with mirrored_strategy.scope():
        _, model_dec, _ = get_video_ED(num_frames, resolution, batch_size_test,
                                       num_slots, slot_size, encode_type,
                                       decode_type, FLAGS.perception_dir)
        model = model_utils.build_IN_LSTM(batch_size,
                                          num_slots,
                                          slot_size,
                                          num_frames=num_frames,
                                          use_camera=False)
        model_test = model_utils.build_IN_LSTM(batch_size_test,
                                               num_slots,
                                               slot_size,
                                               num_frames=num_frames,
                                               use_camera=False)
        optimizer = tf.keras.optimizers.Adam(base_learning_rate, epsilon=1e-08)
        train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        eval_loss = tf.keras.metrics.Mean('eval_loss', dtype=tf.float32)

    # Prepare checkpoint manager.
    global_step = tf.Variable(0,
                              trainable=False,
                              name="global_step",
                              dtype=tf.int64)
    ckpt = tf.train.Checkpoint(network=model,
                               optimizer=optimizer,
                               global_step=global_step)
    ckpt_manager = tf.train.CheckpointManager(checkpoint=ckpt,
                                              directory=FLAGS.model_dir,
                                              max_to_keep=5)
    ckpt.restore(ckpt_manager.latest_checkpoint)
    if ckpt_manager.latest_checkpoint:
        log_fn("Restored from {}".format(ckpt_manager.latest_checkpoint))
    else:
        log_fn("Initializing from scratch.")

    with mirrored_strategy.scope():

        def train_step(batch, model, optimizer, step="train"):
            """Perform a single training step."""
            batch_size = FLAGS.batch_size
            num_frames = FLAGS.num_frames
            num_slots = FLAGS.num_slots
            slot_size = FLAGS.slot_size

            pre_slots = batch['slot']
            B = pre_slots.shape[0]
            pre_slots = tf.nn.sigmoid(pre_slots)
            pre_slots = tf.reshape(
                pre_slots, shape=[-1, num_frames, num_slots, slot_size * 2])
            if step == "train":
                random_noise = tf.random.normal(shape=[B, num_slots])
                random_slots = tf.argsort(random_noise, axis=1)
                pre_slots = tf.gather(pre_slots,
                                      random_slots,
                                      axis=2,
                                      batch_dims=-1)
                del random_slots
            pre_slots_dist = (pre_slots[:, :, :, :slot_size] * 6.0 - 3.0,
                              pre_slots[:, :, :, slot_size:] * 3.0)
            objects = pre_slots_dist[0]
            if step == "train":
                with tf.GradientTape() as tape:
                    objects_2 = model(objects, training=True)
                    objects_2 = tf.roll(objects_2, shift=1, axis=1)
                    loss = tf.math.squared_difference(objects_2, objects)
                    loss = loss[:, 1:, :, :]
                    loss = tf.reduce_sum(loss, axis=[1, 2, 3])
                    train_loss.update_state(loss)
                    loss = tf.nn.compute_average_loss(
                        loss, global_batch_size=batch_size)

                # Get and apply gradients.
                gradients = tape.gradient(loss, model.trainable_weights)
                optimizer.apply_gradients(
                    zip(gradients, model.trainable_weights))

        def data_init(img, mask):
            object_img = tf.expand_dims(img, axis=2) * mask + (1 - mask)
            mask_mean = tf.reduce_sum(mask, axis=[3, 4])
            mask_sum = tf.reduce_sum(mask, axis=2)
            mask_sum = tf.clip_by_value(mask_sum, 0.0, 1.0)
            new_image = img * mask_sum + (1 - mask_sum)
            object_img = tf.reshape(object_img,
                                    shape=[-1] +
                                    object_img.shape.as_list()[3:])
            mask_mean = tf.where(mask_mean > 4, 1.0, 0.0)
            return object_img, new_image, mask_sum, mask_mean

        def test_step(batch, model, model_dec, step="test"):
            """Perform a single training step."""
            num_frames = FLAGS.num_frames
            num_slots = FLAGS.num_slots
            slot_size = FLAGS.slot_size

            pre_slots = batch['slot']
            mask = batch['mask']
            B = pre_slots.shape[0]
            pre_slots = tf.nn.sigmoid(pre_slots)
            pre_slots = tf.reshape(
                pre_slots, shape=[-1, num_frames, num_slots, slot_size * 2])
            random_noise = tf.random.normal(shape=[B, num_slots])
            random_slots = tf.argsort(random_noise, axis=1)
            restore_slots = tf.argsort(random_slots, axis=1)
            mask = tf.gather(mask, random_slots, axis=2, batch_dims=-1)
            pre_slots = tf.gather(pre_slots,
                                  random_slots,
                                  axis=2,
                                  batch_dims=-1)
            pre_slots_dist = (pre_slots[:, :, :, :slot_size] * 6.0 - 3.0,
                              pre_slots[:, :, :, slot_size:] * 3.0)
            objects = pre_slots_dist[0]
            objects_2 = model(objects, training=False)
            objects_2 = tf.roll(objects_2, shift=1, axis=1)
            img = batch['image']
            mask = batch['mask']
            objects = tf.gather(objects, restore_slots, axis=2, batch_dims=-1)
            objects_2 = tf.gather(objects_2,
                                  restore_slots,
                                  axis=2,
                                  batch_dims=-1)
            _, new_image, mask_sum, _ = data_init(img, mask)
            recons, depth = decode_objects(objects, model_dec)
            recons_2, depth_2 = decode_objects(objects_2, model_dec)
            loss2 = compare_object_loss(mask_sum, recons, depth, recons_2,
                                        depth_2)
            loss_all = tf.reduce_sum(loss2[:, 1:], axis=1)
            eval_loss.update_state(loss_all)

        def decode_objects(objects, model_dec):
            (B, F, N, V) = objects.shape
            objects = tf.reshape(objects, shape=[-1, V])
            recons, depth, slots = model_dec(objects)
            recons = tf.reshape(recons,
                                shape=[-1, F, N] + recons.shape.as_list()[1:])
            depth = tf.reshape(depth,
                               shape=[-1, F, N] + depth.shape.as_list()[1:])
            return recons, depth

        def compare_object_loss(mask_sum, recons, depth, recons_2, depth_2):
            DM_factor = -1000.0
            recons, _ = tf.split(recons, [7, 1], axis=2)
            depth, _ = tf.split(depth, [7, 1], axis=2)
            masks = tf.nn.softmax(depth * DM_factor, axis=2)
            masks = masks * tf.expand_dims(mask_sum, axis=2)
            recons_image = tf.reduce_sum(recons * masks, axis=2)
            recons_image = recons_image * mask_sum + 1.0 * (1 - mask_sum)
            recons_2, _ = tf.split(recons_2, [7, 1], axis=2)
            depth_2, _ = tf.split(depth_2, [7, 1], axis=2)
            masks_2 = tf.nn.softmax(depth_2 * DM_factor, axis=2)
            masks_2 = masks_2 * tf.expand_dims(mask_sum, axis=2)
            recons_image2 = tf.reduce_sum(recons_2 * masks_2, axis=2)
            recons_image2 = recons_image2 * mask_sum + 1.0 * (1 - mask_sum)
            loss = tf.math.squared_difference(recons_image2, recons_image)
            return tf.reduce_sum(loss, axis=[2, 3, 4])

    with mirrored_strategy.scope():

        @tf.function
        def distributed_train_step(data, model, optimizer, step="train"):
            mirrored_strategy.run(train_step,
                                  args=(data, model, optimizer, step))

        @tf.function
        def distributed_test_step(data, model, model_dec, step="test"):
            mirrored_strategy.run(test_step,
                                  args=(data, model, model_dec, step))

        start = time.time()
        init_epoch = int(global_step)
        stop = False
        for epoch in range(init_epoch, max_epochs):
            if stop:
                break
            for batch, data in enumerate(train_ds):
                all_step = global_step * 200 + batch + 1
                if all_step > max_steps:
                    stop = True
                    break
                if all_step < warmup_steps:
                    learning_rate = base_learning_rate * tf.cast(
                        all_step, tf.float32) / tf.cast(
                            warmup_steps, tf.float32)
                else:
                    learning_rate = base_learning_rate
                learning_rate = learning_rate * (decay_rate**(tf.cast(
                    all_step, tf.float32) / tf.cast(decay_steps, tf.float32)))
                optimizer.lr = learning_rate.numpy()
                # optimizer_fast.lr = learning_rate_fast.numpy()
                distributed_train_step(data, model, optimizer, step="train")
            model_test.set_weights(model.get_weights())
            for batch, data in enumerate(test_ds):
                distributed_test_step(data, model_test, model_dec, step="test")
            viz.update('train_Loss', epoch,
                       {'scalar': train_loss.result().numpy()})
            viz.update('eval_loss', epoch,
                       {'scalar': eval_loss.result().numpy()})
            log_fn(
                "Epoch: {}, train_Loss: {:.6f}, eval_loss: {:.6f}, Time: {}".
                format(epoch,
                       train_loss.result().numpy(),
                       eval_loss.result().numpy(),
                       datetime.timedelta(seconds=time.time() - start)))
            train_loss.reset_states()
            eval_loss.reset_states()

            global_step.assign_add(1)
            # Save the checkpoint of the model.
            saved_ckpt = ckpt_manager.save()
            log_fn("Saved checkpoint: {}".format(saved_ckpt))


if __name__ == "__main__":
    app.run(main)
