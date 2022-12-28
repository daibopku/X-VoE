import numpy as np
import tensorflow as tf
import keras.layers as layers


class Interaction_Net(layers.Layer):
    """"Interaction Net for Component LSTM """

    def __init__(self,
                 num_slots,
                 slot_dims,
                 hidden_dims=512,
                 cells_dims=2056,
                 buffer_dims=1680,
                 **kwargs):
        super(Interaction_Net, self).__init__()
        """
        Args:
        num_frames: number of past frames for present input
        num_slots:  number of slots for each object
        slot_dims:  dimension of each slot
        hidden_dims: dimension of hidden layer for MLP
        cells_dims: dimension of hidden layer for LSTM
        """

        self.hidden_dims = hidden_dims
        self.num_slots = num_slots
        self.slot_dims = slot_dims
        self.buffer_dims = buffer_dims
        self.MLP_rho = tf.keras.Sequential([
            layers.Dense(hidden_dims, activation='gelu'),
            layers.Dense(hidden_dims, activation='gelu'),
            layers.Dense(hidden_dims, activation='gelu'),
        ])
        self.MLP_lambda = tf.keras.Sequential([
            layers.Dense(hidden_dims, activation='gelu'),
            layers.Dense(hidden_dims, activation='gelu'),
            layers.Dense(hidden_dims, activation='gelu'),
        ])
        self.buffer_linear = tf.keras.Sequential([
            layers.Dense(1680, activation='elu'),
        ])

    def object_buffer(self, z):
        """
        generate object buffer vector of input z
        
        Args:
        z:      (batch_size, num_frames, num_slots, slot_dims)
                latent code for each objects from start to time step t
                num_frames is fixed for z, while z[:, t:, :, :] = 0
                z[:,:t,:,:] is the latent code representing object history 
                (batch_size, num_frames, num_slots, slot_dims)
                
        returns:
        z_b:    (batch_size, num_slots, buffer_dims)
        """
        #batch size
        b = z.shape[0]
        z = tf.transpose(
            z, [0, 2, 3, 1])  #(batch_size, num_slots, slot_dims, num_frames)
        z = tf.reshape(z, [b, self.num_slots, -1
                           ])  #(batch_size, num_slots, slot_dims*num_frames)
        z_b = self.buffer_linear(z)  #(batch_size, num_slots, buffer_dims)
        return z_b

    def call(self, z, cells):
        """
        Args:
        z:      latent code for each objects from start to time step t
                (batch_size, num_frames, num_slots, slot_dims)
                
        cells:  memory cells for each object(K intotal) in LSTM at time step t-1
                shape: (batch_size, num_slots, cell_dims)
                
        returns:
        interactin:(batch_size, num_slots, hidden_dims*2)
        """

        #batch size
        z_b = self.object_buffer(z)  #(batch_size, num_slots, buffer_dims)
        N = cells.shape[1]
        h_i = tf.tile(
            tf.expand_dims(cells, axis=2),
            [1, 1, N, 1])  #(batch_size, num_slots, num_slots, cell_dims)
        h_j = tf.tile(
            tf.expand_dims(cells, axis=1),
            [1, N, 1, 1])  #(batch_size, num_slots, num_slots, cell_dims)
        cell2cell_ij = self.MLP_rho(tf.concat(
            [h_i, h_j],
            axis=-1))  #(batch_size, num_slots, num_slots, cell_dims*2)
        cell2in_ij = self.MLP_lambda(
            tf.concat(
                [h_i, tf.tile(tf.expand_dims(z_b, axis=1), [1, N, 1, 1])],
                axis=-1))
        inter_ij = cell2cell_ij + cell2in_ij
        inter = inter_ij
        sum = tf.reduce_sum(inter, axis=2)
        max = tf.reduce_max(inter, axis=2)
        interaction = tf.concat((sum, max), axis=-1)
        return interaction


class LSTM_Cell(layers.Layer):

    def __init__(self, num_units, **kwargs):
        super().__init__()
        self.num_units = num_units
        self.linear_I = tf.keras.Sequential([
            layers.Dense(num_units, use_bias=True, activation='sigmoid'),
        ])
        self.linear_F = tf.keras.Sequential([
            layers.Dense(num_units, use_bias=True, activation='sigmoid'),
        ])
        self.linear_O = tf.keras.Sequential([
            layers.Dense(num_units, use_bias=True, activation='sigmoid'),
        ])
        self.linear_C = tf.keras.Sequential([
            layers.Dense(num_units, use_bias=True, activation='tanh'),
        ])

    def call(self, inputs, Cell, hidden):
        I = self.linear_I(tf.concat([inputs, hidden], axis=-1))
        F = self.linear_F(tf.concat([inputs, hidden], axis=-1))
        O = self.linear_O(tf.concat([inputs, hidden], axis=-1))
        C_tilda = self.linear_C(tf.concat([inputs, hidden], axis=-1))
        C = F * Cell + I * C_tilda
        H = O * tf.tanh(C)
        return (H, C)


class InteractionLSTM_Cell(layers.Layer):
    """
    RNN cell for Interaction LSTM
    
    latent state: 
    S(t, object_buffer, Cells, hidden_state)                 
    t:              <int32>, representing the time step of the input latent code

    object_buffer:  (batch_size, num_frames, num_slots, latent_dims)
                    memory for object input latent code
                    
    Cells:          (batch_size, num_slots, num_units)
                    hidden memory for LSTM 
                    
    hidden_state:   (batch_size, num_slots, num_units)
                    hidden state for LSTM     
                    
    input:          latent code z:(batch_size, num_frames, num_slots, latent_dim)  
                    representing latent code for objects at t time-step
    """

    def __init__(
        self,
        #  num_frames,
        #  num_slots,
        num_slots,
        slot_dims,
        num_frames=15,
        num_units=2056,
        buffer_dims=1680,
        use_camera=False,
    ):
        super(InteractionLSTM_Cell, self).__init__()
        self.num_slots = num_slots
        self.num_units = num_units
        self.num_frames = num_frames
        self.slot_dims = slot_dims

    @property
    def state_size(self):
        return [
            self.num_slots * self.num_units, self.num_slots * self.num_units
        ]

    @property
    def output_size(self):
        return self.slot_dims

    def build(self, inputs_shape):
        #inputs_shape is batch_size * dim
        #for example: the inputs is batch_size * timestep * dim
        #inputs_shape = [batch_size, dim]
        #we can create some variables in this method, and these variables can be used in call()
        self.num_slots = inputs_shape[2]
        self.latent_dim = inputs_shape[3]
        self.IN = Interaction_Net(self.num_slots, self.latent_dim)
        self.LSTM = LSTM_Cell(self.num_units)
        self.decode = tf.keras.Sequential([
            layers.Dense(self.latent_dim, use_bias=True, activation='linear'),
        ])
        self.built = True

    def call(self, inputs, state):
        # call body
        # how to use previous rnn cell output and state to generate new output and hidden
        # print(state[0])
        Cells_pre = state[0]
        state_pre = state[1]
        Cells_pre = tf.reshape(Cells_pre, [-1, self.num_slots, self.num_units])
        state_pre = tf.reshape(state_pre, [-1, self.num_slots, self.num_units])
        interaction = self.IN(inputs, Cells_pre)
        z_b = self.IN.object_buffer(
            inputs)  #(batch_size, num_slots, buffer_dims)
        X = tf.concat((z_b, interaction),
                      axis=-1)  #(batch_size, num_slot, 2704)
        state_new, Cell_new = self.LSTM(X, Cells_pre, state_pre)
        output = self.decode(state_new)

        Cell_new = tf.reshape(Cell_new, [-1, self.num_slots * self.num_units])
        state_new = tf.reshape(state_new,
                               [-1, self.num_slots * self.num_units])
        return output, (Cell_new, state_new)


class InteractionLSTM(tf.keras.models.Model):
    """
    Interaction LSTM model for dynamics prediction

    input:(batch_size, num_frames, num_slots, slots_dim): latent code of the image provided by perception model
    output: (batch_size, num_frames, num_slots, slots_dim): pred latent code providede by lstm
    """

    def __init__(self,
                 num_slots,
                 slot_dims,
                 num_frames=15,
                 LSTM_units=2056,
                 use_camera=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_frames = num_frames
        self.num_slots = num_slots
        if use_camera:
            self.camera_position_embedding = tf.keras.Sequential([
                layers.Dense(slot_dims, activation='linear'),
            ])
        if use_camera == False:
            self.RNN = tf.keras.layers.RNN(InteractionLSTM_Cell(
                num_slots, slot_dims, num_frames, LSTM_units, use_camera),
                                           return_sequences=True)
        else:
            self.RNN = tf.keras.layers.RNN(InteractionLSTM_Cell(
                num_slots + 1, slot_dims, num_frames, LSTM_units, use_camera),
                                           return_sequences=True)
        self.use_camera = use_camera
        mask = tf.linalg.band_part(tf.ones([num_frames, num_frames]), -1, 0)
        self.mask = tf.reshape(mask,
                               [1, self.num_frames, self.num_frames, 1, 1])

    def call(self, inputs, camera=None):
        """
        Args:
            inputs (batch_size, num_frames, num_slots, slots_dim)     (in our case: [4,15,8,16])
            output: (batch_size, num_frames, num_slots, slots_dim)
        """
        if self.use_camera == True:
            camera = self.camera_position_embedding(camera)
            camera = tf.expand_dims(camera, axis=-2)
            inputs = tf.concat((inputs, camera), axis=-2)
        new_input = tf.expand_dims(inputs, axis=1)
        new_input = tf.tile(new_input, multiples=[1, self.num_frames, 1, 1, 1])
        new_input = new_input * self.mask
        predictions = self.RNN(new_input)
        pred_code = predictions[..., :self.num_slots, :]
        return pred_code


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
    model = InteractionLSTM(8, 32, use_camera=False)
    inputs = tf.random.normal([1, 15, 8, 32])
    camera = tf.random.normal([4, 15, 6])
    predictions = model(inputs)
    print(predictions.shape)
