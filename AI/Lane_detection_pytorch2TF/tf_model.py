import tensorflow as tf
from tf_utils import *
class UNet(tf.keras.layers.Layer):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

    def call(self, x):
        # print(f"Before inconv. x.shape: {x.shape}")
        x1 = self.inc(x)
        # print(f"Before down1. x1.shape: {x1.shape}")
        x2 = self.down1(x1)
        # print(f"Before down2. x2.shape: {x2.shape}")
        x3 = self.down2(x2)
        # print(f"Before down3. x3.shape: {x3.shape}")
        x4 = self.down3(x3)
        # print(f"Before down4. x4.shape: {x4.shape}")
        x5 = self.down4(x4)
        # print(f"Before up1. x5.shape: {x5.shape}")
        x = self.up1(x5, x4)
        # print(f"After up1. x.shape: {x.shape}")
        x = self.up2(x, x3)
        # print(f"After up2. x.shape: {x.shape}")
        x = self.up3(x, x2)
        # print(f"After up3. x.shape: {x.shape}")
        x = self.up4(x, x1)
        # print(f"After up4. x.shape: {x.shape}")
        x = self.outc(x)
        # print(f"After outconv. x.shape: {x.shape}")
        return x

if __name__=="__main__":
    x = tf.random.normal([64, 128, 256, 3], dtype=tf.float32)

    unet = UNet(n_channels=3, n_classes=2)
    pred = unet(x)
    print(pred.shape)