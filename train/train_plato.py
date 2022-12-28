import datetime
import time

from absl import app
from absl import flags
import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
from tensorflow_probability import distributions as tfd
from tensorboardX import SummaryWriter

import build_data.video_read as data_utils
import model.dynamics as dy

FLAGS = flags.FLAGS
flags.DEFINE_string("model_dir", "checkpoint/PLATO/",
                    "Where to save the checkpoints.")
flags.DEFINE_string("tensorboard_dir", "tensorboard/PLATO/",
                    "Where to save the process.")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_integer("batch_size", 20, "Batch size for the model.")
flags.DEFINE_integer("num_frames", 15, "Number of frames in video.")
flags.DEFINE_integer("num_slots", 8, "Number of slots in Slot Attention.")
flags.DEFINE_integer("slot_size", 32, "denth of object slot.")
flags.DEFINE_float("learning_rate", 0.0004, "Learning rate.")
flags.DEFINE_integer("max_epochs", 1, "Number of training epochs.")
flags.DEFINE_integer("max_steps", 10, "Number of training steps.")
flags.DEFINE_integer("warmup_steps", 5000,
                     "Number of warmup steps for the learning rate.")
flags.DEFINE_float("decay_rate", 0.5, "Rate for the learning rate decay.")
flags.DEFINE_integer("decay_steps", 50000,
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


def cal_ari(eval_loss, test_loss):
    delta = 0.0
    Ni = len(eval_loss)
    Nj = len(test_loss)
    relative = tf.reduce_mean(
        (test_loss - eval_loss) / ((test_loss + eval_loss) + 1e-8))
    relative_acc = tf.reduce_mean(
        tf.cast(test_loss - eval_loss > delta, dtype=tf.float32))
    eval_loss = tf.tile(eval_loss[:, None], [1, Nj])
    test_loss = tf.tile(test_loss[None, :], [Ni, 1])
    accurancy = tf.reduce_mean(
        tf.cast(test_loss - eval_loss > delta, dtype=tf.float32))
    return accurancy, relative, relative_acc


def main(argv):
    del argv
    # Hyperparameters of the model.
    batch_size = FLAGS.batch_size
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

    mirrored_strategy = tf.distribute.MirroredStrategy()
    log_fn('Number of devices: {}'.format(
        mirrored_strategy.num_replicas_in_sync))
    num_gpu = mirrored_strategy.num_replicas_in_sync

    viz = TensorboardViz(logdir=FLAGS.tensorboard_dir)

    train_ds = data_utils.debug_iterator(batch_size,
                                         split="train",
                                         shuffle=True)
    train_ds = mirrored_strategy.experimental_distribute_dataset(train_ds)

    with mirrored_strategy.scope():
        model = dy.build_IN_LSTM(batch_size,
                                 num_slots,
                                 slot_size,
                                 num_frames=num_frames,
                                 use_camera=False)
        optimizer = tf.keras.optimizers.Adam(base_learning_rate, epsilon=1e-08)
        train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)

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
            pre_slots = tf.nn.sigmoid(pre_slots)
            pre_slots = tf.reshape(
                pre_slots, shape=[-1, num_frames, num_slots, slot_size * 2])
            B = pre_slots.shape[0]
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
            dist1 = tfd.Normal(pre_slots_dist[0], pre_slots_dist[1])
            objects = dist1.sample()
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
                # gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=1)
                optimizer.apply_gradients(
                    zip(gradients, model.trainable_weights))

    with mirrored_strategy.scope():

        @tf.function
        def distributed_train_step(data, model, optimizer, step="train"):
            mirrored_strategy.run(train_step,
                                  args=(data, model, optimizer, step))

        start = time.time()
        init_epoch = int(global_step)
        for epoch in range(init_epoch, max_epochs):
            for batch, data in enumerate(train_ds):
                all_step = global_step * 390 + batch
                if all_step >= max_steps:
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
            viz.update('train_Loss', epoch,
                       {'scalar': train_loss.result().numpy()})
            log_fn("Epoch: {}, train_Loss: {:.6f}, Time: {}".format(
                epoch,
                train_loss.result().numpy(),
                datetime.timedelta(seconds=time.time() - start)))
            train_loss.reset_states()

            global_step.assign_add(1)
            # Save the checkpoint of the model.
            saved_ckpt = ckpt_manager.save()
            log_fn("Saved checkpoint: {}".format(saved_ckpt))


if __name__ == "__main__":
    app.run(main)
