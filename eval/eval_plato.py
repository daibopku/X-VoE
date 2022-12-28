import datetime
import time

from absl import app
from absl import flags
import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
from tensorflow_probability import distributions as tfd

import build_data.video_read as data_utils
import model.dynamics as dy
import model.perception as model_utils
from utils.metrix import cal_holistic, cal_comparative, cal_relative

FLAGS = flags.FLAGS
flags.DEFINE_string("model_dir", "checkpoint/PLATO/",
                    "Where to save the checkpoints.")
flags.DEFINE_string("perception_dir", "checkpoint/perception/",
                    "Where to save the checkpoints.")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_integer("batch_size", 20, "Batch size for the model.")
flags.DEFINE_integer("batch_size_test", 2, "Batch size for the model test.")
flags.DEFINE_integer("num_frames", 15, "Number of frames in video.")
flags.DEFINE_integer("num_slots", 8, "Number of slots in Slot Attention.")
flags.DEFINE_integer("slot_size", 32, "denth of object slot.")

if logging is None:
    # The logging module may have been unloaded when __del__ is called.
    log_fn = print
else:
    log_fn = logging.warning


def get_video_ED(num_frames, resolution, batch_size, num_slots, slot_size,
                 encode_type, decode_type, file):
    model_pre = model_utils.build_model(resolution,
                                        batch_size * num_slots * num_frames,
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
    tf.random.set_seed(FLAGS.seed)
    resolution = (128, 128)

    decode_type = "SBTD"
    encode_type = "ViT"

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.FILE
    mirrored_strategy = tf.distribute.MirroredStrategy()
    log_fn('Number of devices: {}'.format(
        mirrored_strategy.num_replicas_in_sync))
    num_gpu = mirrored_strategy.num_replicas_in_sync

    # Build dataset iterators, optimizers and model.
    collision_ds = data_utils.debug_iterator(batch_size,
                                             split="collision",
                                             shuffle=False)
    collision_ds = mirrored_strategy.experimental_distribute_dataset(
        collision_ds)
    blocking_ds = data_utils.debug_iterator(batch_size,
                                            split="blocking",
                                            shuffle=False)
    blocking_ds = mirrored_strategy.experimental_distribute_dataset(
        blocking_ds)
    permanence_ds = data_utils.debug_iterator(batch_size,
                                              split="permanence",
                                              shuffle=False)
    permanence_ds = mirrored_strategy.experimental_distribute_dataset(
        permanence_ds)
    continuity_ds = data_utils.debug_iterator(batch_size,
                                              split="continuity",
                                              shuffle=False)
    continuity_ds = mirrored_strategy.experimental_distribute_dataset(
        continuity_ds)
    with mirrored_strategy.scope():
        _, model_dec, _ = get_video_ED(num_frames, resolution, batch_size_test,
                                       num_slots, slot_size, encode_type,
                                       decode_type, FLAGS.perception_dir)
        model = dy.build_IN_LSTM(batch_size,
                                 num_slots,
                                 slot_size,
                                 num_frames=num_frames,
                                 use_camera=False)

    # Prepare checkpoint manager.
    global_step = tf.Variable(0,
                              trainable=False,
                              name="global_step",
                              dtype=tf.int64)
    ckpt = tf.train.Checkpoint(network=model, global_step=global_step)
    ckpt_manager = tf.train.CheckpointManager(checkpoint=ckpt,
                                              directory=FLAGS.model_dir,
                                              max_to_keep=5)
    ckpt.restore(ckpt_manager.latest_checkpoint)
    if ckpt_manager.latest_checkpoint:
        log_fn("Restored from {}".format(ckpt_manager.latest_checkpoint))
    else:
        log_fn("Initializing from scratch.")

    with mirrored_strategy.scope():

        def decode_objects(objects, model_dec):
            (B, F, N, V) = objects.shape
            objects = tf.reshape(objects, shape=[-1, V])
            recons, depth, slots = model_dec(objects)
            recons = tf.reshape(recons,
                                shape=[-1, F, N] + recons.shape.as_list()[1:])
            depth = tf.reshape(depth,
                               shape=[-1, F, N] + depth.shape.as_list()[1:])
            return recons, depth

        def decode_object_loss(new_image, mask_sum, recons, depth):
            sigma = 0.05
            DM_factor = -1000.0
            masks = tf.nn.softmax(depth * DM_factor, axis=2)
            masks = masks * tf.expand_dims(mask_sum, axis=2)
            recons_image = tf.reduce_sum(recons * masks, axis=2)
            recons_image = recons_image * mask_sum + 1.0 * (1 - mask_sum)
            dist = tfd.Normal(recons_image, sigma)
            p_x = dist.log_prob(new_image)
            return -1.0 * tf.reduce_sum(p_x, axis=[2, 3, 4]), recons * masks

        def compare_object_loss(mask_sum, recons, depth, recons_2, depth_2):
            sigma = 0.05
            # new_image = tf.expand_dims(new_image, axis=1)
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
            dist = tfd.Normal(recons_image, sigma)
            p_x = dist.log_prob(recons_image2)
            p_x = p_x[:, :, :, :, :]
            return -1.0 * tf.reduce_sum(p_x, axis=[2, 3, 4])

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

        def test_step(batch, model, step="test"):
            """Perform a single training step."""
            # batch_size = FLAGS.batch_size
            num_frames = FLAGS.num_frames
            num_slots = FLAGS.num_slots
            slot_size = FLAGS.slot_size

            pre_slots = batch['slot']
            mask = batch['mask']
            pre_slots = tf.nn.sigmoid(pre_slots)
            pre_slots = tf.reshape(
                pre_slots, shape=[-1, num_frames, num_slots, slot_size * 2])
            pre_slots_dist = (pre_slots[:, :, :, :slot_size] * 6.0 - 3.0,
                              pre_slots[:, :, :, slot_size:] * 3.0)
            dist1 = tfd.Normal(pre_slots_dist[0], pre_slots_dist[1])
            objects = dist1.sample()
            objects_2 = model(objects)
            objects_2 = tf.roll(objects_2, shift=1, axis=1)
            img = batch['image']
            mask = batch['mask']
            return objects, objects_2, img, mask

        def loss_step(batch, model_dec):
            """Perform a single training step."""
            # batch_size = FLAGS.batch_size
            img = batch['image']
            mask = batch['mask']
            _, new_image, mask_sum, _ = data_init(img, mask)
            objects = batch['objects']
            objects_2 = batch['objects_2']
            recons, depth = decode_objects(objects, model_dec)
            recons_2, depth_2 = decode_objects(objects_2, model_dec)
            loss1, _ = decode_object_loss(new_image, mask_sum, recons, depth)
            loss2 = compare_object_loss(mask_sum, recons, depth, recons_2,
                                        depth_2)
            loss_all = tf.reduce_sum(loss1, axis=1) + tf.reduce_sum(
                loss2[:, 1:], axis=1)
            del mask_sum, recons, depth, recons_2, depth_2
            del objects, objects_2
            return loss1, loss2, loss_all

    with mirrored_strategy.scope():

        @tf.function
        def distributed_test_step(data, model, step="test"):
            per_replica_losses = mirrored_strategy.run(test_step,
                                                       args=(data, model,
                                                             step))
            per_replica_losses = mirrored_strategy.experimental_local_results(
                per_replica_losses)
            objects = tf.concat(
                [per_replica_losses[i][0] for i in range(num_gpu)], axis=0)
            objects_2 = tf.concat(
                [per_replica_losses[i][1] for i in range(num_gpu)], axis=0)
            image = tf.concat(
                [per_replica_losses[i][2] for i in range(num_gpu)], axis=0)
            mask = tf.concat(
                [per_replica_losses[i][3] for i in range(num_gpu)], axis=0)
            return objects, objects_2, image, mask

        @tf.function
        def distributed_loss_step(data, model_dec):
            per_replica_losses = mirrored_strategy.run(loss_step,
                                                       args=(data, model_dec))
            per_replica_losses = mirrored_strategy.experimental_local_results(
                per_replica_losses)
            loss1 = tf.concat(
                [per_replica_losses[i][0] for i in range(num_gpu)], axis=0)
            loss2 = tf.concat(
                [per_replica_losses[i][1] for i in range(num_gpu)], axis=0)
            eval_all = tf.concat(
                [per_replica_losses[i][2] for i in range(num_gpu)], axis=0)
            return loss1, loss2, eval_all

        start = time.time()
        # collision
        feature_data = None
        out_data = None
        for batch, data in enumerate(collision_ds):
            objects, objects_2, image, mask = distributed_test_step(
                data, model, step="test")
            feature = {
                "image": image,
                "mask": mask,
                "objects": objects,
                "objects_2": objects_2,
            }
            feature_data = tf.data.Dataset.from_tensor_slices(feature)
            feature_data = feature_data.batch(batch_size_test,
                                              drop_remainder=True)
            feature_data = feature_data.with_options(options)
            feature_data = mirrored_strategy.experimental_distribute_dataset(
                feature_data)
            for batch, data in enumerate(feature_data):
                loss1, loss2, eval_all = distributed_loss_step(data, model_dec)
                out = {
                    "loss1": loss1,
                    "loss2": loss2,
                    "loss_all": eval_all,
                }
                if out_data is None:
                    out_data = tf.data.Dataset.from_tensor_slices(out)
                else:
                    out_data = out_data.concatenate(
                        tf.data.Dataset.from_tensor_slices(out))
            del feature_data
        out_data = out_data.batch(1000)
        iterator = iter(out_data)
        test = []
        for _ in range(6):
            test.append(next(iterator))
        ds = cal_holistic(test,
                          type_name="Collision",
                          model_name="PLATO",
                          figure_dir="./output/")
        ds = cal_comparative(test,
                             type_name="Collision",
                             model_name="PLATO",
                             figure_dir="./output/")
        ds = cal_relative(test,
                          type_name="Collision",
                          model_name="PLATO",
                          figure_dir="./output/")
        log_fn("Time: {}".format(
            datetime.timedelta(seconds=time.time() - start)))

        # blocking
        feature_data = None
        out_data = None
        for batch, data in enumerate(blocking_ds):
            objects, objects_2, image, mask = distributed_test_step(
                data, model, step="test")
            feature = {
                "image": image,
                "mask": mask,
                "objects": objects,
                "objects_2": objects_2,
            }
            feature_data = tf.data.Dataset.from_tensor_slices(feature)
            feature_data = feature_data.batch(batch_size_test,
                                              drop_remainder=True)
            feature_data = feature_data.with_options(options)
            feature_data = mirrored_strategy.experimental_distribute_dataset(
                feature_data)
            for batch, data in enumerate(feature_data):
                loss1, loss2, eval_all = distributed_loss_step(data, model_dec)
                out = {
                    "loss1": loss1,
                    "loss2": loss2,
                    "loss_all": eval_all,
                }
                if out_data is None:
                    out_data = tf.data.Dataset.from_tensor_slices(out)
                else:
                    out_data = out_data.concatenate(
                        tf.data.Dataset.from_tensor_slices(out))
            del feature_data
        out_data = out_data.batch(1000)
        iterator = iter(out_data)
        test = []
        for _ in range(6):
            test.append(next(iterator))
        ds = cal_holistic(test,
                          type_name="Blocking",
                          model_name="PLATO",
                          figure_dir="./output/")
        ds = cal_comparative(test,
                             type_name="Blocking",
                             model_name="PLATO",
                             figure_dir="./output/")
        ds = cal_relative(test,
                          type_name="Blocking",
                          model_name="PLATO",
                          figure_dir="./output/")
        log_fn("Time: {}".format(
            datetime.timedelta(seconds=time.time() - start)))

        # permanence
        feature_data = None
        out_data = None
        for batch, data in enumerate(permanence_ds):
            objects, objects_2, image, mask = distributed_test_step(
                data, model, step="test")
            feature = {
                "image": image,
                "mask": mask,
                "objects": objects,
                "objects_2": objects_2,
            }
            feature_data = tf.data.Dataset.from_tensor_slices(feature)
            feature_data = feature_data.batch(batch_size_test,
                                              drop_remainder=True)
            feature_data = feature_data.with_options(options)
            feature_data = mirrored_strategy.experimental_distribute_dataset(
                feature_data)
            for batch, data in enumerate(feature_data):
                loss1, loss2, eval_all = distributed_loss_step(data, model_dec)
                out = {
                    "loss1": loss1,
                    "loss2": loss2,
                    "loss_all": eval_all,
                }
                if out_data is None:
                    out_data = tf.data.Dataset.from_tensor_slices(out)
                else:
                    out_data = out_data.concatenate(
                        tf.data.Dataset.from_tensor_slices(out))
            del feature_data
        out_data = out_data.batch(1000)
        iterator = iter(out_data)
        test = []
        for _ in range(4):
            test.append(next(iterator))
        ds = cal_holistic(test,
                          type_name="Permanence",
                          model_name="PLATO",
                          figure_dir="./output/")
        ds = cal_comparative(test,
                             type_name="Permanence",
                             model_name="PLATO",
                             figure_dir="./output/")
        ds = cal_relative(test,
                          type_name="Permanence",
                          model_name="PLATO",
                          figure_dir="./output/")
        log_fn("Time: {}".format(
            datetime.timedelta(seconds=time.time() - start)))

        # continuity
        feature_data = None
        out_data = None
        for batch, data in enumerate(continuity_ds):
            objects, objects_2, image, mask = distributed_test_step(
                data, model, step="test")
            feature = {
                "image": image,
                "mask": mask,
                "objects": objects,
                "objects_2": objects_2,
            }
            feature_data = tf.data.Dataset.from_tensor_slices(feature)
            feature_data = feature_data.batch(batch_size_test,
                                              drop_remainder=True)
            feature_data = feature_data.with_options(options)
            feature_data = mirrored_strategy.experimental_distribute_dataset(
                feature_data)
            for batch, data in enumerate(feature_data):
                loss1, loss2, eval_all = distributed_loss_step(data, model_dec)
                out = {
                    "loss1": loss1,
                    "loss2": loss2,
                    "loss_all": eval_all,
                }
                if out_data is None:
                    out_data = tf.data.Dataset.from_tensor_slices(out)
                else:
                    out_data = out_data.concatenate(
                        tf.data.Dataset.from_tensor_slices(out))
            del feature_data
        out_data = out_data.batch(1000)
        iterator = iter(out_data)
        test = []
        for _ in range(6):
            test.append(next(iterator))
        ds = cal_holistic(test,
                          type_name="Continuity",
                          model_name="PLATO",
                          figure_dir="./output/")
        ds = cal_comparative(test,
                             type_name="Continuity",
                             model_name="PLATO",
                             figure_dir="./output/")
        ds = cal_relative(test,
                          type_name="Continuity",
                          model_name="PLATO",
                          figure_dir="./output/")
        log_fn("Time: {}".format(
            datetime.timedelta(seconds=time.time() - start)))


if __name__ == "__main__":
    app.run(main)
