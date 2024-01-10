import tensorflow as tf
import keras.layers as layers
from tensorflow_probability import distributions as tfd


class Transformer(layers.Layer):

    def __init__(self, num_heads, key_dim, mlp_indim, mlp_outdim, **kwargs):
        super(Transformer, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.mlp_indim = mlp_indim
        self.mlp_outdim = mlp_outdim

        self.multihead = layers.MultiHeadAttention(num_heads=self.num_heads,
                                                   key_dim=self.key_dim)
        self.norm_multihead = layers.LayerNormalization()
        self.mlp = tf.keras.Sequential([
            layers.Dense(self.mlp_indim, activation="relu"),
            layers.Dense(self.mlp_outdim)
        ])
        self.norm_mlp = layers.LayerNormalization()

    def call(self, quary):
        quary_norm = self.norm_multihead(quary)
        quary += self.multihead(quary_norm, quary_norm)
        quary_norm = self.norm_mlp(quary)
        quary += self.mlp(quary_norm)
        return quary


def unstack_and_split(x, num_channels=3):
    masks = x[..., 3:4]
    channels = x
    return channels, masks


class VariationalAutoEncoder(layers.Layer):

    def __init__(self, resolution, slot_size, **kwargs):
        super(VariationalAutoEncoder, self).__init__(**kwargs)
        self.resolution = resolution
        self.slot_size = slot_size

        self.encoder_cnn = tf.keras.Sequential([
            layers.Conv2D(64,
                          kernel_size=5,
                          strides=2,
                          padding="SAME",
                          activation=tf.nn.leaky_relu),
            layers.Conv2D(64,
                          kernel_size=5,
                          strides=2,
                          padding="SAME",
                          activation=tf.nn.leaky_relu),
            layers.Conv2D(64,
                          kernel_size=5,
                          strides=2,
                          padding="SAME",
                          activation=tf.nn.leaky_relu),
            layers.Conv2D(64,
                          kernel_size=5,
                          strides=2,
                          padding="SAME",
                          activation=tf.nn.leaky_relu)
        ],
                                               name="encoder_cnn")
        self.flatten = layers.Flatten()

        # self.encoder_pos = SoftPositionEmbed(64, self.resolution)

        # self.layer_norm = layers.LayerNormalization()
        self.mlp = tf.keras.Sequential([
            layers.Dense(512, activation=tf.nn.leaky_relu),
            layers.Dense(slot_size * 2)
        ],
                                       name="feedforward")

    def call(self, image):
        # `image` has shape: [batch_size, width, height, num_channels].
        x = self.encoder_cnn(image)
        x = self.flatten(x)
        pre_slots = self.mlp(x)
        # slots = tf.sigmoid(pre_slots)
        return pre_slots


class ViTAutoEncoder(layers.Layer):

    def __init__(self, resolution, slot_size, num_channels, **kwargs):
        super(ViTAutoEncoder, self).__init__(**kwargs)
        self.resolution = resolution
        self.slot_size = slot_size
        self.num_channels = num_channels
        self.flatten = layers.Flatten()

        self.patch_size = (16, 16)
        assert self.resolution[0] % self.patch_size[
            0] == 0 and self.resolution[1] % self.patch_size[1] == 0, (
                'resolution must divided by patch')
        self.grid_size = (int(self.resolution[0] / self.patch_size[0]),
                          int(self.resolution[1] / self.patch_size[1]))

        x = tf.linspace(-1, 1, self.patch_size[1])
        y = tf.linspace(-1, 1, self.patch_size[0])
        x_grid, y_grid = tf.meshgrid(x, y)
        self.grid = tf.concat(
            [tf.expand_dims(x_grid, axis=2),
             tf.expand_dims(y_grid, axis=2)],
            axis=-1)
        self.encode_size = self.num_channels * self.grid_size[
            0] * self.grid_size[1]
        self.dense_pre = layers.Dense(self.encode_size,
                                      activation=None,
                                      name="dense_pre")
        self.pos_enc = layers.Dense(self.encode_size, activation=None)

        self.slots_mu = self.add_weight(initializer="glorot_uniform",
                                        shape=[1, 1, self.encode_size],
                                        dtype=tf.float32,
                                        name="slots_mu")
        num_head = 8
        assert self.encode_size % num_head == 0, (
            "encode_size must divided by num_head")
        self.transformer = tf.keras.Sequential([
            Transformer(num_head, int(self.encode_size / num_head),
                        self.encode_size * 4, self.encode_size)
            for _ in range(6)
        ],
                                               name="transformer")
        self.mlp = tf.keras.Sequential([
            layers.Dense(512, activation=tf.nn.leaky_relu),
            layers.Dense(slot_size * 2)
        ],
                                       name="feedforward")

    def call(self, image):
        # `image` has shape: [batch_size, width, height, num_channels].
        B = image.shape[0]
        # `x` has shape: [batch_size, patch_size_w, grid_size_x,patch_size_h, grid_size_h, num_channels].
        x = tf.reshape(image,
                       shape=[
                           B, self.patch_size[0], self.grid_size[0],
                           self.patch_size[1], self.grid_size[1],
                           self.num_channels
                       ])
        x = tf.transpose(x, perm=[0, 1, 3, 2, 4, 5])
        x = tf.reshape(x,
                       shape=[B, self.patch_size[0], self.patch_size[1], -1])
        x = self.dense_pre(x)
        x = x + self.pos_enc(self.grid)
        x = tf.reshape(x,
                       shape=[B, self.patch_size[0] * self.patch_size[1], -1])
        x = tf.concat([tf.tile(self.slots_mu, multiples=[B, 1, 1]), x], axis=1)
        x = self.transformer(x)
        x = x[:, 0, :]
        pre_slots = self.mlp(x)
        return pre_slots


class SampleNorm(layers.Layer):

    def __init__(self, slot_size, **kwargs) -> None:
        super(SampleNorm, self).__init__(**kwargs)
        self.slot_size = slot_size

    def call(self, pre_slots):
        slots = tf.nn.sigmoid(pre_slots)
        x_mean = slots[:, :self.slot_size] * 6.0 - 3.0
        x_sig = slots[:, self.slot_size:] * 3.0
        dist = tfd.Normal(x_mean, x_sig)
        x = dist.sample()
        kl_z = tfd.kl_divergence(dist, tfd.Normal(0., 1.))
        kl_z = tf.reduce_sum(kl_z, axis=-1)
        return x, kl_z


class Decoder(layers.Layer):

    def __init__(self, slot_size, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.slot_size = slot_size
        self.decoder_cnn = tf.keras.Sequential([
            layers.Conv2DTranspose(64,
                                   5,
                                   strides=(2, 2),
                                   padding="SAME",
                                   activation=tf.nn.leaky_relu),
            layers.Conv2DTranspose(64,
                                   5,
                                   strides=(2, 2),
                                   padding="SAME",
                                   activation=tf.nn.leaky_relu),
            layers.Conv2DTranspose(64,
                                   5,
                                   strides=(2, 2),
                                   padding="SAME",
                                   activation=tf.nn.leaky_relu),
            layers.Conv2DTranspose(64,
                                   5,
                                   strides=(2, 2),
                                   padding="SAME",
                                   activation=tf.nn.leaky_relu),
            layers.Conv2DTranspose(64,
                                   5,
                                   strides=(2, 2),
                                   padding="SAME",
                                   activation=tf.nn.leaky_relu),
            layers.Conv2DTranspose(64,
                                   5,
                                   strides=(2, 2),
                                   padding="SAME",
                                   activation=tf.nn.leaky_relu),
            layers.Conv2DTranspose(64,
                                   5,
                                   strides=(2, 2),
                                   padding="SAME",
                                   activation=tf.nn.leaky_relu),
            layers.Conv2DTranspose(64,
                                   5,
                                   strides=(1, 1),
                                   padding="SAME",
                                   activation=tf.nn.leaky_relu),
            layers.Conv2DTranspose(
                4, 3, strides=(1, 1), padding="SAME", activation=None)
        ],
                                               name="decoder_cnn")
        self.depth_mask = self.add_weight(initializer="glorot_uniform",
                                          shape=[1],
                                          dtype=tf.float32,
                                          name="depth_mask")

    def call(self, slots):
        # Spatial broadcast decoder.
        x = tf.reshape(slots, shape=[-1, 1, 1, self.slot_size])
        # `x` has shape: [batch_size, width, height, slot_size].
        x = self.decoder_cnn(x)
        # Undo combination of slot and batch dimension; split alpha masks.
        depths = x[..., 3:4]
        recons = x
        masks = tf.nn.relu(self.depth_mask) * depths
        # `recons` has shape: [batch_size, width, height, num_channels].
        # `masks` has shape: [batch_size, width, height, 1].
        return recons, masks, slots


class SBDecoder(layers.Layer):

    def __init__(self, resolution, slot_size, **kwargs):
        super(SBDecoder, self).__init__(**kwargs)
        self.resolution = resolution
        self.slot_size = slot_size

        x = tf.linspace(-1, 1, resolution[1])
        y = tf.linspace(-1, 1, resolution[0])
        x_grid, y_grid = tf.meshgrid(x, y)
        self.x_grid = tf.cast(tf.reshape(x_grid,
                                         [1, resolution[0], resolution[1], 1]),
                              dtype=tf.float32)
        self.y_grid = tf.cast(tf.reshape(y_grid,
                                         [1, resolution[0], resolution[1], 1]),
                              dtype=tf.float32)

        self.decoder_cnn = tf.keras.Sequential([
            layers.Conv2D(slot_size * 2,
                          kernel_size=1,
                          padding="SAME",
                          activation=tf.nn.leaky_relu),
            layers.Conv2D(slot_size * 2,
                          kernel_size=1,
                          padding="SAME",
                          activation=tf.nn.leaky_relu),
            layers.Conv2D(slot_size * 2,
                          kernel_size=1,
                          padding="SAME",
                          activation=tf.nn.leaky_relu),
            layers.Conv2D(4, 1, padding="SAME", activation=None)
        ],
                                               name="decoder_cnn")
        self.depth_mask = self.add_weight(initializer="glorot_uniform",
                                          shape=[1],
                                          dtype=tf.float32,
                                          name="depth_mask")

    def call(self, slots):
        batch_size = slots.shape[0]
        # Spatial broadcast decoder.

        x = tf.reshape(slots, shape=[-1, 1, 1, self.slot_size])
        # `x` has shape: [batch_size, width, height, slot_size].
        x = tf.tile(x,
                    multiples=[1, self.resolution[0], self.resolution[1], 1])
        x = tf.concat([
            x,
            tf.tile(self.x_grid, multiples=[batch_size, 1, 1, 1]),
            tf.tile(self.y_grid, multiples=[batch_size, 1, 1, 1])
        ],
                      axis=-1)
        x = self.decoder_cnn(x)
        # Undo combination of slot and batch dimension; split alpha masks.
        depths = x[..., 3:4]
        recons = x
        masks = tf.abs(self.depth_mask) * depths
        # `recons` has shape: [batch_size, width, height, num_channels].
        # `masks` has shape: [batch_size, width, height, 1].
        return recons, masks, slots


class SBTDecoder(layers.Layer):

    def __init__(self, resolution, slot_size, **kwargs):
        super(SBTDecoder, self).__init__(**kwargs)
        self.resolution = resolution
        self.slot_size = slot_size

        self.decoder_initial_size = (8, 8)
        self.decoder_cnn = tf.keras.Sequential([
            layers.Conv2DTranspose(self.slot_size,
                                   5,
                                   strides=(2, 2),
                                   padding="SAME",
                                   activation="relu"),
            layers.Conv2DTranspose(self.slot_size,
                                   5,
                                   strides=(2, 2),
                                   padding="SAME",
                                   activation="relu"),
            layers.Conv2DTranspose(self.slot_size,
                                   5,
                                   strides=(2, 2),
                                   padding="SAME",
                                   activation="relu"),
            layers.Conv2DTranspose(self.slot_size,
                                   5,
                                   strides=(2, 2),
                                   padding="SAME",
                                   activation="relu"),
            layers.Conv2DTranspose(self.slot_size,
                                   5,
                                   strides=(1, 1),
                                   padding="SAME",
                                   activation="relu"),
            layers.Conv2DTranspose(
                4, 3, strides=(1, 1), padding="SAME", activation=None)
        ],
                                               name="decoder_cnn")

        x = tf.linspace(-1, 1, self.decoder_initial_size[1])
        y = tf.linspace(-1, 1, self.decoder_initial_size[0])
        x_grid, y_grid = tf.meshgrid(x, y)
        self.grid = tf.concat(
            [tf.expand_dims(x_grid, axis=2),
             tf.expand_dims(y_grid, axis=2)],
            axis=-1)
        self.pos_enc = layers.Dense(slot_size, activation=None)
        self.depth_mask = self.add_weight(initializer="glorot_uniform",
                                          shape=[1],
                                          dtype=tf.float32,
                                          name="depth_mask")

    def call(self, slots):
        # Spatial broadcast decoder.

        x = tf.reshape(slots, shape=[-1, 1, 1, self.slot_size])
        # `x` has shape: [batch_size, width, height, slot_size].
        x = tf.tile(x,
                    multiples=[
                        1, self.decoder_initial_size[0],
                        self.decoder_initial_size[1], 1
                    ])
        x = x + tf.expand_dims(self.pos_enc(self.grid), axis=0)
        x = self.decoder_cnn(x)
        # Undo combination of slot and batch dimension; split alpha masks.
        depths = x[..., 3:4]
        recons = x
        masks = tf.abs(self.depth_mask) * depths
        # `recons` has shape: [batch_size, width, height, num_channels].
        # `masks` has shape: [batch_size, width, height, 1].
        return recons, masks, slots


def build_model(resolution,
                batch_size,
                num_channels=3,
                slot_size=64,
                decode_type="SBD",
                encode_type="default",
                **kwargs):
    """Build keras model."""
    if decode_type == "SBD":
        model_dec = SBDecoder(resolution,
                              slot_size,
                              name="ObjectDecoder",
                              **kwargs)
    elif decode_type == "SBTD":
        model_dec = SBTDecoder(resolution,
                               slot_size,
                               name="ObjectDecoder",
                               **kwargs)
    else:
        model_dec = Decoder(slot_size, name="ObjectDecoder", **kwargs)

    if encode_type == "ViT":
        model_enc = ViTAutoEncoder(resolution,
                                   slot_size,
                                   num_channels,
                                   name="ObjectEncoder",
                                   **kwargs)
    else:
        model_enc = VariationalAutoEncoder(resolution,
                                           slot_size,
                                           name="ObjectEncoder",
                                           **kwargs)
    image = tf.keras.Input(list(resolution) + [num_channels], batch_size)
    pre_slots = model_enc(image)
    slots, kl_z = SampleNorm(slot_size, name="SampleNorm")(pre_slots)
    recons, masks, slots = model_dec(slots)
    model = tf.keras.Model(inputs=image, outputs=[recons, masks, slots, kl_z])
    return model
