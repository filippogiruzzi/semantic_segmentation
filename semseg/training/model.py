import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Conv2D, MaxPool2D, Conv2DTranspose, concatenate


class UNet(Model):
    def __init__(self, params=None, is_training=False):
        super(UNet, self).__init__()

        self.is_training = is_training
        self.input_size = params['input_size']
        self.n_classes = params['n_classes']

        ##### Encoder
        # Block 1
        self.conv1_1 = Conv2D(64, 3, activation='relu', padding='same', name='conv1_1')
        self.conv1_2 = Conv2D(64, 3, activation='relu', padding='same', name='conv1_2')
        self.pool1 = MaxPool2D(2, 2, name='pool1')

        # Block 2
        self.conv2_1 = Conv2D(128, 3, activation='relu', padding='same', name='conv2_1')
        self.conv2_2 = Conv2D(128, 3, activation='relu', padding='same', name='conv2_2')
        self.pool2 = MaxPool2D(2, 2, name='pool2')

        # Block 3
        self.conv3_1 = Conv2D(256, 3, activation='relu', padding='same', name='conv3_1')
        self.conv3_2 = Conv2D(256, 3, activation='relu', padding='same', name='conv3_2')
        self.conv3_3 = Conv2D(256, 3, activation='relu', padding='same', name='conv3_3')
        self.pool3 = MaxPool2D(2, 2, name='pool3')

        # Block 4
        self.conv4_1 = Conv2D(512, 3, activation='relu', padding='same', name='conv4_1')
        self.conv4_2 = Conv2D(512, 3, activation='relu', padding='same', name='conv4_2')
        self.conv4_3 = Conv2D(512, 3, activation='relu', padding='same', name='conv4_3')
        self.pool4 = MaxPool2D(2, 2, name='pool4')

        # Block 5
        self.conv5_1 = Conv2D(512, 3, activation='relu', padding='same', name='conv5_1')
        self.conv5_2 = Conv2D(512, 3, activation='relu', padding='same', name='conv5_2')
        self.conv5_3 = Conv2D(512, 3, activation='relu', padding='same', name='conv5_3')

        ##### Decoder
        self.dec_convt1 = Conv2DTranspose(256, 2, 2, padding='same', name='dec_convt1')
        self.dec_conv1 = Conv2D(256, 3, activation='relu', padding='same', name='dec_conv1')
        self.dec_conv2 = Conv2D(256, 3, activation='relu', padding='same', name='dec_conv2')

        self.dec_convt2 = Conv2DTranspose(128, 2, 2, padding='same', name='dec_convt2')
        self.dec_conv3 = Conv2D(128, 3, activation='relu', padding='same', name='dec_conv3')
        self.dec_conv4 = Conv2D(128, 3, activation='relu', padding='same', name='dec_conv4')

        self.dec_convt3 = Conv2DTranspose(64, 2, 2, padding='same', name='dec_convt3')
        self.dec_conv5 = Conv2D(64, 3, activation='relu', padding='same', name='dec_conv5')
        self.dec_conv6 = Conv2D(64, 3, activation='relu', padding='same', name='dec_conv6')

        self.dec_convt4 = Conv2DTranspose(32, 2, 2, padding='same', name='dec_convt4')
        self.dec_conv7 = Conv2D(32, 3, activation='relu', padding='same', name='dec_conv7')
        self.dec_conv8 = Conv2D(32, 3, activation='relu', padding='same', name='dec_conv8')

        self.dec_conv9 = Conv2D(self.n_classes, 1, activation='linear', padding='same', name='dec_conv9')

    def call(self, inputs, training=None, mask=None):
        img_input = inputs['img_input']

        with tf.name_scope('encoder'):
            with tf.name_scope('block1'):
                x = self.conv1_1(img_input)
                x = self.conv1_2(x)
                out_block1 = x
                x = self.pool1(x)
            with tf.name_scope('block2'):
                x = self.conv2_1(x)
                x = self.conv2_2(x)
                out_block2 = x
                x = self.pool2(x)
            with tf.name_scope('block3'):
                x = self.conv3_1(x)
                x = self.conv3_2(x)
                x = self.conv3_3(x)
                out_block3 = x
                x = self.pool3(x)
            with tf.name_scope('block4'):
                x = self.conv4_1(x)
                x = self.conv4_2(x)
                x = self.conv4_3(x)
                out_block4 = x
                x = self.pool4(x)
            with tf.name_scope('block5'):
                x = self.conv5_1(x)
                x = self.conv5_2(x)
                x = self.conv5_3(x)
                out_block5 = x

        with tf.name_scope('decoder'):
            x = self.dec_convt1(out_block5)
            x = concatenate([x, out_block4], axis=-1)
            x = self.dec_conv1(x)
            x = self.dec_conv2(x)

            x = self.dec_convt2(x)
            x = concatenate([x, out_block3], axis=-1)
            x = self.dec_conv3(x)
            x = self.dec_conv4(x)

            x = self.dec_convt3(x)
            x = concatenate([x, out_block2], axis=-1)
            x = self.dec_conv5(x)
            x = self.dec_conv6(x)

            x = self.dec_convt4(x)
            x = concatenate([x, out_block1], axis=-1)
            x = self.dec_conv7(x)
            x = self.dec_conv8(x)

            output = self.dec_conv9(x)
        return output
