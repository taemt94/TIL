import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, MaxPool2D, UpSampling2D, Conv2DTranspose


class double_conv(tf.keras.layers.Layer):
    def __init__(self, out_ch):
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
    def __init__(self, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(out_ch)

    def call(self, x):
        x = self.conv(x)
        return x

class down(tf.keras.layers.Layer):
    def __init__(self, out_ch):
        super(down, self).__init__()
        self.mpconv = Sequential([
            MaxPool2D(pool_size=2),
            double_conv(out_ch)
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

        self.conv = double_conv(out_ch)

    def call(self, x_dec, x_enc):
        x_dec = self.up(x_dec)
        # print(x_dec.shape)
        diffX = x_dec.shape[1] - x_enc.shape[1]
        diffY = x_dec.shape[2] - x_enc.shape[2]
        ### paddings 설명 추가
        ### 홀수 입력 시에는 안되는 것 아닌지?
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
    def __init__(self, out_ch):
        super(outconv, self).__init__()
        self.conv = Conv2D(filters=out_ch, kernel_size=1)

    def call(self, x):
        # print("Shape before conv:", x.shape)
        x = self.conv(x)
        # print("Shape after  conv:", x.shape)
        return x