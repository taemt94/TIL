import tensorflow as tf
from tf_utils import *
import config
import time

def limit_gpu(gb):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    num_gpu = 1
    memory_limit = 1024 * gb
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(gpus[num_gpu - 1], [
                tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)])
            print("Use {} GPU limited {}MB memory".format(num_gpu, memory_limit))
        except RuntimeError as e:
            print(e)

    else:
        print('GPU is not available')

# limit_gpu(8)
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# if len(physical_devices) > 0:
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)
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
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x

class UNet_ConvLSTM(tf.keras.Model):
    # def __init__(self, n_channels, n_classes):
    #     super(UNet_ConvLSTM, self).__init__()
    #     self.inc = inconv(n_channels, 64)
    #     self.down1 = down(64, 128)
    #     self.down2 = down(128, 256)
    #     self.down3 = down(256, 512)
    #     self.down4 = down(512, 512)
    #     self.up1 = up(1024, 256)
    #     self.up2 = up(512, 128)
    #     self.up3 = up(256, 64)
    #     self.up4 = up(128, 64)

    #     self.outc = outconv(64, n_classes)
    #     self.convlstm = ConvLSTM(input_size=(8, 16),
    #                              input_dim=512,
    #                              hidden_dim=[512, 512],
    #                              kernel_size=(3, 3),
    #                              num_layers=2,
    #                              batch_first=True,
    #                              bias=True,
    #                              return_all_layers=False)
    def __init__(self, n_channels, n_classes):
        super(UNet_ConvLSTM, self).__init__()
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
        self.convlstm = ConvLSTM(input_size=(8, 16),
                                 input_dim=512,
                                 hidden_dim=[512, 512],
                                 kernel_size=(3, 3),
                                 num_layers=2,
                                 batch_first=True,
                                 bias=True,
                                 return_all_layers=False)
    
    def call(self, x):
        # print(f"call {x.shape}")
        # x = tf.unstack(x, axis=1)
        
        x = tf.transpose(x, [1,0,2,3,4])
        # data = tf.tensor()
        for i, item in enumerate(x):
            # print(i, item.shape)
            x1 = self.inc(item)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            # print(f"x5: {x5.shape}")
            x5 = tf.expand_dims(x5, axis=1)
            # print(f"x5: {x5.shape}")
            if i == 0:
                data = x5
            else:
                data = tf.concat([data, x5], axis=1)
            # data.append(x5)
            # print(len(data))
        # print(f"len: {len(data)}")
        # data = tf.concat(data, axis=1)
        # print(f"After concat: {data.shape}")
        data = tf.transpose(data, perm=[0, 1, 4, 2, 3])
        # print(f"Before convlstm: {data.shape}")
        lstm, _ = self.convlstm(data)
        test = lstm[-1][-1,:, :, :, :]
        # print(f"After convlstm: {test.shape}")
        test = tf.transpose(test, perm=[0, 2, 3, 1])
        # print(f"After transpose: {test.shape}")

        x = self.up1(test, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x, test

def generate_model(args):
    
    # use_cuda = args.cuda and tf.test.is_gpu_available()
    # device = tf.device("gpu" if use_cuda else "cpu")

    assert args.model in [ 'UNet-ConvLSTM', 'SegNet-ConvLSTM', 'UNet', 'SegNet']
    # if args.model == 'SegNet-ConvLSTM':
    #     model = SegNet_ConvLSTM()
    # elif args.model == 'SegNet':
    #     model = SegNet()
    if args.model == 'UNet-ConvLSTM':
        model = UNet_ConvLSTM(config.img_channel, config.class_num)
    elif args.model == 'UNet':
        model = UNet(config.img_channel, config.class_num)
    return model

class TestModel(tf.keras.Model):
    def __init__(self):
        super(TestModel, self).__init__()
        time.sleep(60)
    def call(self, x):
        return x

if __name__=="__main__":
    
    x = tf.random.normal([64, 5, 128, 256, 3], dtype=tf.float32)

    # unet = UNet(n_channels=3, n_classes=2)
    # pred = unet(x)
    # print(pred.shape)

    unetConvLSTM = UNet_ConvLSTM(n_channels=3, n_classes=2)
    # pred, test = unetConvLSTM(x)
    # print(pred.shape)
    # print(test.shape)
    # testmodel = TestModel()