import tensorflow as tf
from model.perception import SBDecoder, SBTDecoder, Decoder, VariationalAutoEncoder, ViTAutoEncoder, SampleNorm
from model.explain import FastThink
from model.dynamics import InteractionLSTM


def build_percept_model(resolution,
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


def build_IN_LSTM(batch_size,
                  num_slots,
                  slot_dims,
                  num_frames=15,
                  LSTM_units=2056,
                  use_camera=False):
    """
    build Interaction LSTM model
    """
    lstm = InteractionLSTM(num_slots, slot_dims, num_frames, LSTM_units,
                           use_camera)
    latent_code = tf.keras.Input([num_frames, num_slots, slot_dims],
                                 batch_size)
    outputs = lstm(latent_code)
    model = tf.keras.Model(inputs=latent_code, outputs=outputs)
    return model


if __name__ == '__main__':
    explain_model = build_fast_model(64, 15, 8, 32)
    inputs = tf.random.normal([64, 15, 8, 32])
    mask = tf.ones([64, 15, 8, 1])
    predictions = explain_model((inputs, mask))
    print(predictions.shape)
    dynamic_model = InteractionLSTM(8, 32, use_camera=False)
    inputs = tf.random.normal([1, 15, 8, 32])
    camera = tf.random.normal([4, 15, 6])
    predictions = dynamic_model(inputs)
    print(predictions.shape)