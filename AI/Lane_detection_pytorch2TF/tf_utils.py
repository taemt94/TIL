import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, MaxPool2D, UpSampling2D, Conv2DTranspose


class double_conv(tf.keras.layers.Layer):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = Sequential([
            Conv2D(filters=out_ch, kernel_size=3, padding='same'),
            BatchNormalization(),
            ReLU(),
            Conv2D(filters=out_ch, kernel_size=3, padding='same'),
            BatchNormalization(),
            ReLU()
        ])

    def call(self, x):
        # print(x.shape)
        x = self.conv(x)
        # print(x.shape)
        return x

class inconv(tf.keras.layers.Layer):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def call(self, x):
        x = self.conv(x)
        return x

class down(tf.keras.layers.Layer):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = Sequential([
            MaxPool2D(pool_size=2),
            double_conv(in_ch, out_ch)
        ])

    def call(self, x):
        # print(x.shape)
        x = self.mpconv(x)
        # print(x.shape)
        return x


class up(tf.keras.layers.Layer):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        if bilinear:
            self.up = UpSampling2D(size=(2, 2))
        else:
            self.up = Conv2DTranspose(filters=in_ch // 2, kernel_size=2, strides=2)

        self.conv = double_conv(in_ch, out_ch)

    def call(self, x_dec, x_enc):
        x_dec = self.up(x_dec)
        # print(x_dec.shape)
        diffX = x_dec.shape[1] - x_enc.shape[1]
        diffY = x_dec.shape[2] - x_enc.shape[2]
        paddings = [[0, 0],
                    [diffX // 2, int(diffX / 2)],
                    [diffY // 2, int(diffY / 2)],
                    [0, 0]]
        x_enc = tf.pad(x_enc, paddings=paddings)
        x = tf.concat([x_enc, x_dec], axis=-1)
        # print(x_enc.shape, x_dec.shape)
        # print(x.shape)
        x = self.conv(x)
        # print(x.shape)
        return x
class outconv(tf.keras.layers.Layer):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = Conv2D(filters=out_ch, kernel_size=1)

    def call(self, x):
        # print("Shape before conv:", x.shape)
        x = self.conv(x)
        # print("Shape after  conv:", x.shape)
        return x

## TF ConvLSTMCell
class ConvLSTMCell(tf.keras.layers.Layer):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """
        
        super(ConvLSTMCell, self).__init__()
        
        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = Conv2D(filters=4*self.hidden_dim, kernel_size=self.kernel_size, padding=self.padding, use_bias=self.bias)

    def call(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = tf.concat([input_tensor, h_cur], axis=-1)

        combined_conv = self.conv(combined)

        cc_i, cc_f, cc_o, cc_g = tf.split(value=combined_conv, num_or_size_splits=self.hidden_dim, axis=1)
        i = tf.sigmoid(cc_i)
        f = tf.sigmoid(cc_f)
        o = tf.sigmoid(cc_o)
        g = tf.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * tf.tanh(c_next)
        # print(h_next.shape, c_next.shape)
        return h_next, c_next
    
    def init_hidden(self, batch_size):
        return (tf.zeros([batch_size, self.hidden_dim, self.height, self.width], dtype=tf.float32),
                tf.zeros([batch_size, self.hidden_dim, self.height, self.width], dtype=tf.float32))

## TF ConvSLTM class
class ConvLSTM(tf.keras.layers.Layer):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers, batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        ## Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        # print(f"kernel_size: {kernel_size}, hidden_dim: {hidden_dim}")
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError("Inconsistent list length.")
        
        self.height, self.width = input_size

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        self.cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            self.cell_list.append(ConvLSTMCell(input_size=(self.height, self.width),
                                               input_dim=cur_input_dim,
                                               hidden_dim=self.hidden_dim[i],
                                               kernel_size=self.kernel_size[i],
                                               bias=self.bias))

    def call(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            ## (timesteps, batch_size, channels, height, width) -> (batch_size, timesteps, channels, height, width)
            input_tensor = tf.transpose(input_tensor, perm=[1, 0, 2, 3, 4])
        
        ## Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=input_tensor.shape[0])
        
        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.shape[1]
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]            
            output_inner = []
            # print(f"Layer {layer_idx + 1} starts.") 
            for t in range(seq_len):
                # print()
                # print(f"Seqeunce {t + 1} starts.")
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :], cur_state=[h, c])
                # print(f"h.shape: {h.shape} c.shape: {c.shape}")
                output_inner.append(h)
            
            layer_output = tf.stack(output_inner, axis=1)
            cur_layer_input = layer_output

            layer_output = tf.transpose(layer_output, perm=[1, 0, 2, 3, 4])
            # print(f"layer_output.shape: {layer_output.shape}")
            # print()
            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]
        
        return layer_output_list, last_state_list
    
    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
            # for j in range(2):
            #     print(f"init_state[{i}].shape: {init_states[i][j].shape}")
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(_kernel_size):
        if not (isinstance(_kernel_size, tuple) or (isinstance(_kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')
    
    @staticmethod
    def _extend_for_multilayer(param, _num_layers):
        if not isinstance(param, list):
            param = [param] * _num_layers
        return param