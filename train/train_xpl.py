import datetime
import time

from absl import app
from absl import flags
import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
from tensorflow_probability import distributions as tfd
from tensorboardX import SummaryWriter

import build_data.video_read as data_utils
import model.model as model_parse
import model.dynamics as dy

FLAGS = flags.FLAGS
flags.DEFINE_string("model_dir", "checkpoint/XPL/",
                    "Where to save the checkpoints.")
flags.DEFINE_string("tensorboard_dir", "tensorboard/XPL/",
                    "Where to save the process.")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_integer("batch_size", 20, "Batch size for the model.")
flags.DEFINE_integer("num_frames", 15, "Number of frames in video.")
flags.DEFINE_integer("num_slots", 8, "Number of slots in Slot Attention.")
flags.DEFINE_integer("slot_size", 32, "denth of object slot.")
flags.DEFINE_integer("sample_steps_num", 100, "number of samples step.")
flags.DEFINE_float("alpha", 0.01, "multiplier of reason loss")
flags.DEFINE_float("learning_rate", 0.0004, "Learning rate.")
flags.DEFINE_float("learning_sample", 0.04, "Sample Learning rate.")
flags.DEFINE_float("learning_fast", 0.0004, "Fast Learning rate.")
flags.DEFINE_integer("max_epochs", 1, "Number of training steps.")
flags.DEFINE_integer("max_steps", 10, "Number of training steps.")
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


def main(argv):
    del argv
    # Hyperparameters of the model.
    batch_size = FLAGS.batch_size
    num_frames = FLAGS.num_frames
    num_slots = FLAGS.num_slots
    slot_size = FLAGS.slot_size
    sample_steps_num = FLAGS.sample_steps_num
    alpha = FLAGS.alpha
    base_learning_rate = FLAGS.learning_rate
    max_epochs = FLAGS.max_epochs
    max_steps = FLAGS.max_steps
    warmup_steps = FLAGS.warmup_steps
    decay_rate = FLAGS.decay_rate
    decay_steps = FLAGS.decay_steps
    learning_sample = FLAGS.learning_sample
    learning_fast = FLAGS.learning_fast
    tf.random.set_seed(FLAGS.seed)
    patch_size = (1, 3, 5, 15)

    mirrored_strategy = tf.distribute.MirroredStrategy()
    log_fn('Number of devices: {}'.format(
        mirrored_strategy.num_replicas_in_sync))
    num_gpu = mirrored_strategy.num_replicas_in_sync

    viz = TensorboardViz(logdir=FLAGS.tensorboard_dir)

    # Build dataset iterators, optimizers and model.
    train_ds = data_utils.debug_iterator(batch_size,
                                         split="train",
                                         shuffle=True)
    train_ds = mirrored_strategy.experimental_distribute_dataset(train_ds)

    with mirrored_strategy.scope():
        optimizer = tf.keras.optimizers.Adam(base_learning_rate, epsilon=1e-08)
        optimizer_sample = tf.keras.optimizers.Adam(learning_sample,
                                                    epsilon=1e-08)
        optimizer_fast = tf.keras.optimizers.Adam(learning_fast, epsilon=1e-08)
        optimizer_new = tf.keras.optimizers.Adam(learning_fast, epsilon=1e-08)

        model_sample = model_parse.build_sample_model(num_frames,
                                                      patch_size,
                                                      int(batch_size /
                                                          num_gpu),
                                                      num_slots,
                                                      slot_size,
                                                      initial=True)
        model_reason = dy.build_IN_LSTM(batch_size,
                                        num_slots,
                                        slot_size,
                                        num_frames=num_frames,
                                        use_camera=False)
        model_fast = model_parse.build_fast_model(batch_size, num_frames,
                                                  num_slots, slot_size)
        model_new = model_parse.build_fast_model(batch_size, num_frames,
                                                 num_slots, slot_size)
        reason_loss = tf.keras.metrics.Mean('reason_loss', dtype=tf.float32)
        fast_loss = tf.keras.metrics.Mean('fast_loss', dtype=tf.float32)
        new_loss = tf.keras.metrics.Mean('new_loss', dtype=tf.float32)

    model_sample.get_layer("parsetree").re_init()

    # Prepare checkpoint manager.
    global_step = tf.Variable(0,
                              trainable=False,
                              name="global_step",
                              dtype=tf.int64)
    ckpt = tf.train.Checkpoint(network=model_reason,
                               optimizer=optimizer,
                               global_step=global_step)
    ckpt_F = tf.train.Checkpoint(
        network=model_fast,
        optimizer=optimizer_fast,
    )
    ckpt_N = tf.train.Checkpoint(
        network=model_new,
        optimizer=optimizer_new,
    )
    ckpt_manager = tf.train.CheckpointManager(checkpoint=ckpt,
                                              directory=FLAGS.model_dir,
                                              max_to_keep=5)
    ckpt.restore(ckpt_manager.latest_checkpoint)
    ckpt_manager_F = tf.train.CheckpointManager(checkpoint=ckpt_F,
                                                directory=FLAGS.model_dir +
                                                "/fast",
                                                max_to_keep=5)
    ckpt_F.restore(ckpt_manager_F.latest_checkpoint)
    ckpt_manager_N = tf.train.CheckpointManager(checkpoint=ckpt_N,
                                                directory=FLAGS.model_dir +
                                                "/new",
                                                max_to_keep=5)
    ckpt_N.restore(ckpt_manager_N.latest_checkpoint)
    if ckpt_manager.latest_checkpoint:
        log_fn("Restored from {}".format(ckpt_manager.latest_checkpoint))
    else:
        log_fn("Initializing from scratch.")
    if ckpt_manager_F.latest_checkpoint:
        log_fn("Restored from {}".format(ckpt_manager_F.latest_checkpoint))
    else:
        log_fn("Initializing from scratch.")
    if ckpt_manager_N.latest_checkpoint:
        log_fn("Restored from {}".format(ckpt_manager_N.latest_checkpoint))
    else:
        log_fn("Initializing from scratch.")

    with mirrored_strategy.scope():

        def train_step(batch,
                       model_sample,
                       model_reason,
                       model_fast,
                       model_new,
                       optimizer_sample,
                       optimizer_reason,
                       optimizer_fast,
                       optimizer_new,
                       multi_tupe,
                       sample_steps_num,
                       step="train"):
            """Perform a single training step."""
            batch_size = FLAGS.batch_size
            num_frames = FLAGS.num_frames
            num_slots = FLAGS.num_slots
            slot_size = FLAGS.slot_size
            (multi1, multi2) = multi_tupe

            pre_slots = batch['slot']
            mask = batch['mask']
            B = pre_slots.shape[0]
            pre_slots = tf.nn.sigmoid(pre_slots)
            pre_slots = tf.reshape(
                pre_slots, shape=[-1, num_frames, num_slots, slot_size * 2])
            if step == "train":
                random_noise = tf.random.normal(shape=[B, num_slots])
                random_slots = tf.argsort(random_noise, axis=1)
                mask = tf.gather(mask, random_slots, axis=2, batch_dims=-1)
                pre_slots = tf.gather(pre_slots,
                                      random_slots,
                                      axis=2,
                                      batch_dims=-1)
            mask_mean = tf.reduce_sum(mask, axis=[3, 4])
            mask_mean = tf.where(mask_mean > 4, 1.0, 0.0)
            mask_mean = mask_mean + tf.cast(
                tf.reduce_sum(mask_mean, axis=1, keepdims=True) < 1,
                tf.float32)
            mask_axes = tf.concat([
                tf.zeros([1, num_frames, 1, 1]),
                tf.ones([1, num_frames, num_slots - 1, 1])
            ],
                                  axis=2)
            mask_axes = tf.tile(mask_axes, [B, 1, 1, 1])
            pre_slots_dist = (pre_slots[:, :, :, :slot_size] * 6.0 - 3.0,
                              pre_slots[:, :, :, slot_size:] * 3.0)
            dist1 = tfd.Normal(pre_slots_dist[0], pre_slots_dist[1])
            objects_pre = dist1.sample()
            objects_init = model_fast((objects_pre, mask_mean), training=False)

            for _ in range(sample_steps_num):
                # Get the prediction of the models and compute the loss.
                with tf.GradientTape() as tape:
                    model_sample.trainable = True
                    model_reason.trainable = False
                    model_fast.trainable = False
                    _, objects = model_sample(objects_pre, training=True)
                    objects = objects + objects_init
                    objects = pre_slots_dist[0] * mask_mean + objects * (
                        1 - mask_mean)
                    objects_2 = model_reason(objects)
                    objects_2 = objects + objects_2
                    objects_2 = tf.roll(objects_2, shift=1, axis=1)
                    loss1 = tf.math.squared_difference(objects_2, objects)
                    loss1 = loss1[:, 1:, :, :]
                    loss2 = tf.math.squared_difference(objects_init, objects)
                    loss1 = tf.reduce_sum(loss1, axis=[1, 2, 3])
                    loss2 = tf.reduce_sum(loss2, axis=[1, 2, 3])
                    loss = loss1 * multi1 + loss2 * multi2
                # Get and apply gradients.
                gradients = tape.gradient(loss, model_sample.trainable_weights)
                optimizer_sample.apply_gradients(
                    zip(gradients, model_sample.trainable_weights),
                    experimental_aggregate_gradients=False)

            _, objects = model_sample(objects_pre, training=False)
            objects = objects + objects_init
            objects = pre_slots_dist[0] * mask_mean + objects * (1 - mask_mean)
            if step == "train":
                with tf.GradientTape() as tape:
                    model_sample.trainable = False
                    model_reason.trainable = True
                    model_fast.trainable = False
                    (B, F, N, V) = objects.shape
                    objects_2 = model_reason(objects, training=True)
                    objects_2 = objects + objects_2
                    objects_2 = tf.roll(objects_2, shift=1, axis=1)
                    loss = tf.math.squared_difference(objects_2, objects)
                    loss = loss[:, 1:, :, :]
                    loss = tf.reduce_sum(loss, axis=[1, 2, 3])
                    reason_loss.update_state(loss)
                    loss = tf.nn.compute_average_loss(
                        loss, global_batch_size=batch_size)

                # Get and apply gradients.
                gradients = tape.gradient(loss, model_reason.trainable_weights)
                optimizer_reason.apply_gradients(
                    zip(gradients, model_reason.trainable_weights))

                with tf.GradientTape() as tape:
                    model_sample.trainable = False
                    model_reason.trainable = False
                    model_fast.trainable = True
                    objects_init = model_fast((objects_pre, mask_mean),
                                              training=True)
                    objects_init = pre_slots_dist[
                        0] * mask_mean + objects_init * (1 - mask_mean)
                    loss = tf.math.squared_difference(objects_init, objects)
                    loss = tf.reduce_sum(loss, axis=[1, 2, 3])
                    fast_loss.update_state(loss)
                    loss = tf.nn.compute_average_loss(
                        loss, global_batch_size=batch_size)

                # Get and apply gradients.
                gradients = tape.gradient(loss, model_fast.trainable_weights)
                optimizer_fast.apply_gradients(
                    zip(gradients, model_fast.trainable_weights))

                with tf.GradientTape() as tape:
                    model_sample.trainable = False
                    model_reason.trainable = False
                    model_fast.trainable = False
                    model_new.trainable = True
                    objects_init = model_new((objects, mask_axes),
                                             training=True)
                    loss = tf.math.squared_difference(objects_init, objects)
                    loss = tf.reduce_sum(loss, axis=[1, 2, 3])
                    new_loss.update_state(loss)
                    loss = tf.nn.compute_average_loss(
                        loss, global_batch_size=batch_size)

                # Get and apply gradients.
                gradients = tape.gradient(loss, model_new.trainable_weights)
                # gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=1)
                optimizer_new.apply_gradients(
                    zip(gradients, model_new.trainable_weights))

    with mirrored_strategy.scope():

        @tf.function
        def distributed_train_step(data,
                                   model_sample,
                                   model_reason,
                                   model_fast,
                                   model_new,
                                   optimizer_sample,
                                   optimizer_reason,
                                   optimizer_fast,
                                   optimizer_new,
                                   multi_tupe,
                                   sample_steps_num,
                                   step="train"):
            mirrored_strategy.run(
                train_step,
                args=(data, model_sample, model_reason, model_fast, model_new,
                      optimizer_sample, optimizer_reason, optimizer_fast,
                      optimizer_new, multi_tupe, sample_steps_num, step))

        start = time.time()
        init_epoch = int(global_step)
        multi_tupe = (1.0, alpha)
        for epoch in range(init_epoch, max_epochs):
            for batch, data in enumerate(train_ds):
                all_step = global_step * 208 + batch
                if all_step >= max_steps:
                    break
                if all_step < warmup_steps:
                    learning_rate_fast = learning_fast * tf.cast(
                        all_step, tf.float32) / tf.cast(
                            warmup_steps, tf.float32)
                    learning_rate_reason = base_learning_rate * tf.cast(
                        all_step, tf.float32) / tf.cast(
                            warmup_steps, tf.float32)
                else:
                    learning_rate_fast = learning_fast
                    learning_rate_reason = base_learning_rate
                learning_rate_fast = learning_rate_fast * (
                    decay_rate**(tf.cast(all_step, tf.float32) /
                                 tf.cast(decay_steps, tf.float32)))
                learning_rate_reason = learning_rate_reason * (
                    decay_rate**(tf.cast(all_step, tf.float32) /
                                 tf.cast(decay_steps, tf.float32)))
                model_sample.get_layer("parsetree").re_init()
                for var in optimizer_sample.variables():
                    var.assign(tf.zeros_like(var))
                optimizer.lr = learning_rate_reason.numpy()
                optimizer_fast.lr = learning_rate_fast.numpy()
                optimizer_new.lr = learning_rate_fast.numpy()
                distributed_train_step(data,
                                       model_sample,
                                       model_reason,
                                       model_fast,
                                       model_new,
                                       optimizer_sample,
                                       optimizer,
                                       optimizer_fast,
                                       optimizer_new,
                                       multi_tupe,
                                       sample_steps_num,
                                       step="train")
            viz.update('reason_loss', epoch,
                       {'scalar': reason_loss.result().numpy()})
            viz.update('fast_loss', epoch,
                       {'scalar': fast_loss.result().numpy()})
            viz.update('new_loss', epoch,
                       {'scalar': new_loss.result().numpy()})
            log_fn("Epoch: {}, Time: {}".format(
                epoch, datetime.timedelta(seconds=time.time() - start)))
            log_fn(
                "learning loss : reason_loss: {:.6f}, fast_loss: {:.6f}, new_loss: {:.6f}"
                .format(reason_loss.result().numpy(),
                        fast_loss.result().numpy(),
                        new_loss.result().numpy()))
            reason_loss.reset_states()
            fast_loss.reset_states()
            new_loss.reset_state()

            global_step.assign_add(1)
            # Save the checkpoint of the model.
            saved_ckpt = ckpt_manager.save()
            log_fn("Saved checkpoint: {}".format(saved_ckpt))
            saved_ckpt_F = ckpt_manager_F.save()
            log_fn("Saved checkpoint F: {}".format(saved_ckpt_F))
            saved_ckpt_N = ckpt_manager_N.save()
            log_fn("Saved checkpoint N: {}".format(saved_ckpt_N))


if __name__ == "__main__":
    app.run(main)
