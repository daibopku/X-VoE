import tensorflow as tf
import keras.layers as layers


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


class SampleParseTree(layers.Layer):
    def __init__(self,
                 batch_size,
                 num_frames,
                 patch_size,
                 num_slots,
                 slot_size,
                 initial=False,
                 trainable=True,
                 name=None,
                 dtype=None,
                 dynamic=False,
                 **kwargs):
        super(SampleParseTree, self).__init__(trainable, name, dtype, dynamic,
                                              **kwargs)
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.patch_size = patch_size
        self.num_patch = len(patch_size)
        self.num_slots = num_slots
        self.slot_size = slot_size
        self.initial = initial

        for i in patch_size:
            assert self.num_frames % i == 0, (
                'num_frames must divided by patch_size')

    def build(self, slot_size):
        super().build(slot_size)
        """build parse tree parameters."""
        # objects parameter in parse tree, size: [batch_size, num_frames, num_slots, slot_size]
        # initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
        initializer = tf.keras.initializers.get("zeros")
        self.object_list = []
        for i, patch in enumerate(self.patch_size):
            self.object_list.append(
                self.add_weight(name="objects_{}".format(i),
                                shape=[
                                    self.batch_size, patch, self.num_slots,
                                    self.slot_size
                                ],
                                dtype=tf.float32,
                                initializer=initializer,
                                trainable=True))

    def re_init(self, std=1.):
        initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=std)
        # object parameter in parse tree, size: [batch_size, num_frames, num_slots, slot_size]
        if self.initial:
            for i, patch in enumerate(self.patch_size):
                self.object_list[i].assign(
                    tf.zeros(shape=[
                        self.batch_size, patch, self.num_slots, self.slot_size
                    ]))
        else:
            for i, patch in enumerate(self.patch_size):
                self.object_list[i].assign(
                    initializer(shape=[
                        self.batch_size, patch, self.num_slots, self.slot_size
                    ]))

    def call(self, inputs):
        objects = tf.zeros(
            [self.batch_size, self.num_frames, self.num_slots, self.slot_size])
        for i, patch in enumerate(self.patch_size):
            objects += tf.tile(
                self.object_list[i],
                multiples=[1, int(self.num_frames / patch), 1, 1])
        return inputs, objects


class FastThink(layers.Layer):
    """Slot Attention module."""
    def __init__(self,
                 num_frames,
                 num_slots,
                 slot_size,
                 num_transformer,
                 mlp_hidden_size,
                 epsilon=1e-8,
                 **kwargs):
        super(FastThink, self).__init__(**kwargs)
        self.num_frames = num_frames
        self.num_slots = num_slots
        self.slot_size = slot_size
        self.mlp_hidden_size = mlp_hidden_size
        # self.mlp_layers = mlp_layers
        self.epsilon = epsilon

        x = tf.linspace(-1, 1, self.num_slots)
        y = tf.linspace(-1, 1, self.num_frames)
        x_grid, y_grid = tf.meshgrid(x, y)
        self.grid = tf.concat(
            [tf.expand_dims(x_grid, axis=2),
             tf.expand_dims(y_grid, axis=2)],
            axis=-1)
        self.mask_layer = self.add_weight(
            initializer="glorot_uniform",
            shape=[1, 1, 1, 1, self.mlp_hidden_size],
            dtype=tf.float32,
            name="mask_layer")
        self.pos_enc = layers.Dense(self.mlp_hidden_size, activation=None)
        self.pos_dec = layers.Dense(self.mlp_hidden_size, activation=None)
        self.encoder = layers.Dense(self.mlp_hidden_size,
                                    activation=None,
                                    name="fast_enc")

        num_heads = 8
        assert (self.mlp_hidden_size) % num_heads == 0, (
            "encode_size must divided by num_head")
        key_dim = int(self.mlp_hidden_size / num_heads)
        self.transformer = tf.keras.Sequential([
            Transformer(
                num_heads,
                key_dim,
                self.mlp_hidden_size,
                self.mlp_hidden_size,
            ) for _ in range(num_transformer)
        ],
                                               name="transformer")

        self.decoder = layers.Dense(self.slot_size,
                                    activation=None,
                                    name="fast_dec")

    def call(self, inputs, mask_mean):
        (B, _, _, _) = inputs.shape
        # `inputs` has shape [batch_size, num_frames, num_slots, inputs_size].
        x = self.encoder(inputs)  # [batch_size, num_frames, num_slots, dim].
        x = x * mask_mean + tf.tile(
            self.mask_layer[:, 0, :, :, :],
            [1, self.num_frames, self.num_slots, 1]) * (1 - mask_mean)
        x = tf.reshape(
            x, [B, self.num_frames, self.num_slots, self.mlp_hidden_size])
        x = x + self.pos_enc(
            self.grid)  # [batch_size, num_frames, num_slots, dim].
        x = tf.reshape(x,
                       shape=[
                           -1, self.num_frames * self.num_slots,
                           self.mlp_hidden_size
                       ])  # [batch_size, num_frames*num_slots, dim].
        x = self.transformer(x)  # [batch_size, num_frames*num_slots, dim].
        x = tf.reshape(x,
                       shape=[
                           -1, self.num_frames, self.num_slots,
                           self.mlp_hidden_size
                       ])  # [batch_size, num_frames, num_slots, dim].
        x = self.decoder(
            x
        )  # [batch_size, num_frames, num_slots, slot_size + slot_size_vel].
        object_init = tf.reshape(
            x, [B, self.num_frames, self.num_slots, self.slot_size])
        object_init = inputs * mask_mean + object_init * (1 - mask_mean)
        return object_init


def build_sample_model(num_frames,
                       patch_size,
                       batch_size,
                       num_slots,
                       slot_size,
                       initial=True,
                       **kwargs):
    objects = tf.keras.Input([num_frames, num_slots, slot_size])
    image_out, parament_dist = SampleParseTree(batch_size,
                                               num_frames,
                                               patch_size,
                                               num_slots,
                                               slot_size,
                                               initial=initial,
                                               name="parsetree",
                                               **kwargs)(objects)
    model = tf.keras.Model(inputs=objects, outputs=(image_out, parament_dist))
    return model


def build_fast_model(batch_size,
                     num_frames,
                     num_slots,
                     slot_size,
                     num_transformer=6,
                     mlp_hidden_size=256,
                     **kwargs):
    pre_slots = tf.keras.Input([num_frames, num_slots, slot_size], batch_size)
    mask_mean = tf.keras.Input([num_frames, num_slots, 1], batch_size)
    loss = FastThink(num_frames,
                     num_slots,
                     slot_size,
                     num_transformer,
                     mlp_hidden_size,
                     name="lossreasoning",
                     **kwargs)(pre_slots, mask_mean)
    model = tf.keras.Model(inputs=[pre_slots, mask_mean], outputs=loss)
    return model


def build_new_model(batch_size,
                    num_frames,
                    num_slots,
                    slot_size,
                    num_transformer=6,
                    mlp_hidden_size=256,
                    **kwargs):
    pre_slots = tf.keras.Input([num_frames, num_slots, slot_size], batch_size)
    mask_mean = tf.keras.Input([num_frames, num_slots, 1], batch_size)
    loss = FastThink(num_frames,
                     num_slots,
                     slot_size,
                     num_transformer,
                     mlp_hidden_size,
                     name="lossreasoning",
                     **kwargs)(pre_slots, mask_mean)
    model = tf.keras.Model(inputs=[pre_slots, mask_mean], outputs=loss)
    return model


if __name__ == '__main__':
    patch_size = (1, 3, 5, 15)
    model = build_sample_model(15, patch_size, 16, 8, 32, True)
    inputs = tf.random.normal([64, 15, 8, 32])
    mask = tf.ones([64, 15, 8, 1])
    predictions = model(inputs)
    print(predictions[1].shape)
    model = build_new_model(64, 15, 8, 32)
    inputs = tf.random.normal([64, 15, 8, 32])
    mask = tf.ones([64, 15, 8, 1])
    predictions = model((inputs, mask))
    print(predictions.shape)