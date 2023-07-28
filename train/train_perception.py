import datetime
import time

from absl import app
from absl import flags
import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
from tensorflow_probability import distributions as tfd
from tensorboardX import SummaryWriter

import build_data.image_read as data_utils
import model.model as model_utils

FLAGS = flags.FLAGS
flags.DEFINE_string("model_dir", "XPL/Perception/",
                    "Where to save the checkpoints.")
flags.DEFINE_string("tensorboard_dir", "tensorboard/Perception/",
                    "Where to save the process.")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_string("encode_type", "ViT", "type of encoder model")
flags.DEFINE_string("decode_type", "SBTD", "type of decoder model")
flags.DEFINE_integer("batch_size", 300, "Batch size for the model.")
flags.DEFINE_integer("num_frames", 1, "Number of frames in video.")
flags.DEFINE_integer("num_slots", 8, "Number of slots in Slot Attention.")
flags.DEFINE_integer("slot_size", 32, "denth of slot.")
flags.DEFINE_float("DM_factor", -1000.0, "factor of depth to mask")
flags.DEFINE_float("beta", 0.5, "multiplier of slots kl")
flags.DEFINE_float("sigma", 0.05, "std of image pixel")
flags.DEFINE_float("learning_rate", 0.0004, "Learning rate.")
flags.DEFINE_integer("max_epochs", 118, "Number of training steps.")
flags.DEFINE_integer("warmup_steps", 2000,
                     "Number of warmup steps for the learning rate.")
flags.DEFINE_float("decay_rate", 0.5, "Rate for the learning rate decay.")
flags.DEFINE_integer("decay_steps", 100000,
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


def data_init(batch):
    img = batch['image']
    mask = batch['mask']
    object_img = tf.expand_dims(img, axis=2) * mask + (1 - mask)
    mask_sum = tf.reduce_sum(mask, axis=2)
    mask_sum = tf.clip_by_value(mask_sum, 0.0, 1.0)
    new_image = img * mask_sum + (1 - mask_sum)
    object_img = tf.reshape(object_img,
                            shape=[-1] + object_img.shape.as_list()[3:])
    return object_img, new_image, mask_sum


def main(argv):
    del argv
    # Hyperparameters of the model.
    encode_type = FLAGS.encode_type
    decode_type = FLAGS.decode_type
    batch_size = FLAGS.batch_size
    num_frames = FLAGS.num_frames
    num_slots = FLAGS.num_slots
    # mlp_layers = FLAGS.mlp_layers
    slot_size = FLAGS.slot_size
    # num_iterations = FLAGS.num_iterations
    base_learning_rate = FLAGS.learning_rate
    max_epochs = FLAGS.max_epochs
    warmup_steps = FLAGS.warmup_steps
    decay_rate = FLAGS.decay_rate
    decay_steps = FLAGS.decay_steps
    tf.random.set_seed(FLAGS.seed)
    resolution = (128, 128)

    mirrored_strategy = tf.distribute.MirroredStrategy()
    log_fn('Number of devices: {}'.format(
        mirrored_strategy.num_replicas_in_sync))

    viz = TensorboardViz(logdir=FLAGS.tensorboard_dir)

    # Build dataset iterators, optimizers and model.
    train_size = 80000 * 15
    test_size = 20000 * 15
    train_ds, test_ds = data_utils.load_data(batch_size,
                                             train_size=train_size,
                                             test_size=test_size,
                                             shuffle=True)
    train_ds = mirrored_strategy.experimental_distribute_dataset(train_ds)
    test_ds = mirrored_strategy.experimental_distribute_dataset(test_ds)

    global_step = tf.Variable(0,
                              trainable=False,
                              name="global_step",
                              dtype=tf.int64)
    with mirrored_strategy.scope():
        model = model_utils.build_model(resolution,
                                        batch_size * num_slots * num_frames,
                                        num_channels=4,
                                        slot_size=slot_size,
                                        decode_type=decode_type,
                                        encode_type=encode_type)
        optimizer = tf.keras.optimizers.Adam(base_learning_rate, epsilon=1e-08)

        ckpt = tf.train.Checkpoint(network=model,
                                   optimizer=optimizer,
                                   global_step=global_step)

        # Define our metrics
        train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
    ckpt_manager = tf.train.CheckpointManager(checkpoint=ckpt,
                                              directory=FLAGS.model_dir,
                                              max_to_keep=5)
    ckpt.restore(ckpt_manager.latest_checkpoint)
    if ckpt_manager.latest_checkpoint:
        log_fn("Restored from {}".format(ckpt_manager.latest_checkpoint))
    else:
        log_fn("Initializing from scratch.")

    with mirrored_strategy.scope():

        def train_step(batch, model, optimizer):
            """Perform a single training step."""
            beta = FLAGS.beta
            sigma = FLAGS.sigma
            num_frames = FLAGS.num_frames
            num_slots = FLAGS.num_slots
            DM_factor = FLAGS.DM_factor

            # Get the prediction of the models and compute the loss.
            with tf.GradientTape() as tape:
                image, new_image, mask_sum = data_init(batch)
                preds = model(image, training=True)
                recons, depth, slots, kl_z = preds
                recons = tf.reshape(recons,
                                    shape=[-1, num_frames, num_slots] +
                                    recons.shape.as_list()[1:])
                depth = tf.reshape(depth,
                                   shape=[-1, num_frames, num_slots] +
                                   depth.shape.as_list()[1:])
                kl_z = tf.reshape(kl_z, shape=[-1, num_frames, num_slots])
                masks = tf.nn.softmax(depth * DM_factor, axis=2)
                masks = masks * tf.expand_dims(mask_sum, axis=2)
                dist = tfd.Normal(recons, sigma)
                p_x = dist.log_prob(tf.expand_dims(new_image, axis=2))
                p_x *= masks
                p_x = tf.reduce_sum(p_x, axis=2, keepdims=True)
                p_x = p_x + (1.0 - tf.expand_dims(mask_sum, axis=2))
                p_x = tf.reduce_sum(p_x, axis=[3, 4, 5])
                loss_value = -1.0 * tf.reduce_mean(
                    p_x, axis=[1, 2]) + beta * tf.reduce_mean(kl_z,
                                                              axis=[1, 2])
                train_loss.update_state(loss_value)
                loss_value = tf.nn.compute_average_loss(
                    loss_value, global_batch_size=batch_size)
                del recons, masks, slots  # Unused.

            # Get and apply gradients.
            gradients = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(gradients, model.trainable_weights))

        def test_step(batch, model):
            """Perform a single training step."""

            # Get the prediction of the models and compute the loss.
            beta = FLAGS.beta
            sigma = FLAGS.sigma
            num_frames = FLAGS.num_frames
            num_slots = FLAGS.num_slots
            DM_factor = FLAGS.DM_factor

            # Get the prediction of the models and compute the loss.
            image, new_image, mask_sum = data_init(batch)
            preds = model(image, training=True)
            recons, depth, slots, kl_z = preds
            recons = tf.reshape(recons,
                                shape=[-1, num_frames, num_slots] +
                                recons.shape.as_list()[1:])
            depth = tf.reshape(depth,
                               shape=[-1, num_frames, num_slots] +
                               depth.shape.as_list()[1:])
            kl_z = tf.reshape(kl_z, shape=[-1, num_frames, num_slots])
            masks = tf.nn.softmax(depth * DM_factor, axis=2)
            masks = masks * tf.expand_dims(mask_sum, axis=2)
            dist = tfd.Normal(recons, sigma)
            p_x = dist.log_prob(tf.expand_dims(new_image, axis=2))
            p_x *= masks
            p_x = tf.reduce_sum(p_x, axis=2, keepdims=True)
            p_x = p_x + (1.0 - tf.expand_dims(mask_sum, axis=2))
            p_x = tf.reduce_sum(p_x, axis=[3, 4, 5])
            loss_value = -1.0 * tf.reduce_mean(
                p_x, axis=[1, 2]) + beta * tf.reduce_mean(kl_z, axis=[1, 2])
            test_loss.update_state(loss_value)
            del recons, masks, slots  # Unused.

    with mirrored_strategy.scope():

        @tf.function
        def distributed_train_step(data, model, optimizer):
            return mirrored_strategy.run(train_step,
                                         args=(data, model, optimizer))

        @tf.function
        def distributed_test_step(data, model):
            return mirrored_strategy.run(test_step, args=(data, model))

        start = time.time()
        init_epoch = int(global_step)
        for epoch in range(init_epoch, max_epochs):
            for batch, data in enumerate(train_ds):
                all_step = global_step * 4000 + batch
                # Learning rate warm-up.
                if all_step < warmup_steps:
                    learning_rate = base_learning_rate * tf.cast(
                        all_step, tf.float32) / tf.cast(
                            warmup_steps, tf.float32)
                else:
                    learning_rate = base_learning_rate
                learning_rate = learning_rate * (decay_rate**(tf.cast(
                    all_step, tf.float32) / tf.cast(decay_steps, tf.float32)))
                optimizer.lr = learning_rate.numpy()
                distributed_train_step(data, model, optimizer)

                # # Log the training loss.
                if not (batch + 1) % 400:
                    log_fn("Epoch: {}, Train_Loss: {:.6f}".format(
                        global_step.numpy(),
                        train_loss.result().numpy()))

            for batch, data in enumerate(test_ds):
                distributed_test_step(data, model)

            viz.update('train_loss', epoch,
                       {'scalar': train_loss.result().numpy()})
            viz.update('eval_loss', epoch,
                       {'scalar': test_loss.result().numpy()})
            log_fn("Epoch: {}, Train_Loss: {:.6f}, Eval_Loss: {:.6f}".format(
                global_step.numpy(),
                train_loss.result().numpy(),
                test_loss.result().numpy()))
            log_fn("Time: {}".format(
                datetime.timedelta(seconds=time.time() - start)))
            train_loss.reset_states()
            test_loss.reset_states()

            global_step.assign_add(1)
            # Save the checkpoint of the model.
            saved_ckpt = ckpt_manager.save()
            log_fn("Saved checkpoint: {}".format(saved_ckpt))


if __name__ == "__main__":
    app.run(main)
