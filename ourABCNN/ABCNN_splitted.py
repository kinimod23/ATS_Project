import tensorflow as tf
import numpy as np


############################################################################
# ToDo:
#    implement conv and deconv as seperate models

class ABCNN_conv():
    def __init__(self, s, w, l2_reg, d0=300, di=52, num_layers=2):
        """
        Implmenentaion of ABCNNs
        (https://arxiv.org/pdf/1512.05193.pdf)
        :param s: sentence length
        :param w: filter width
        :param l2_reg: L2 regularization coefficient
        :param d0: dimensionality of word embedding(default: 300)
        :param di: The number of convolution kernels (default: 50)
        :param num_layers: The number of convolution layers.
        """

        self.x1 = tf.placeholder(tf.float32, shape=[None, d0, s], name="x1")
        self.x2 = tf.placeholder(tf.float32, shape=[None, d0, s], name="x2")
        self.y1 = tf.placeholder(tf.int32, shape=[None], name="y1")

        # zero padding to inputs for wide convolution
        def pad_for_wide_conv(x):
            return tf.pad(x, np.array([[0, 0], [0, 0], [w - 1, w - 1], [0, 0]]), "CONSTANT", name="pad_wide_conv")

        def cos_sim(v1, v2, axis=None):
            norm1 = tf.sqrt(tf.reduce_sum(tf.square(v1), axis=axis))
            norm2 = tf.sqrt(tf.reduce_sum(tf.square(v2), axis=axis))
            dot_products = tf.reduce_sum(v1 * v2, axis=axis, name="cos_sim")
            return dot_products / (norm1 * norm2)

        def w_pool(variable_scope, x):
            with tf.variable_scope(variable_scope + "-w_pool"):
                w_ap = tf.layers.average_pooling2d(
                    inputs=x, pool_size=(1, w),
                    strides=1, padding="VALID", name="w_ap")
                return w_ap

        def all_pool(variable_scope, x):
            with tf.variable_scope(variable_scope + "-all_pool"):
                if variable_scope.startswith("input"):
                    pool_width, d = s, d0
                else: pool_width, d = s + w - 1, di
                all_ap = tf.layers.average_pooling2d(
                    inputs=x, pool_size=(1, pool_width),
                    strides=1, padding="VALID", name="all_ap")
                all_ap_reshaped = tf.reshape(all_ap, [-1, d])
                return all_ap_reshaped

        def convolution(name_scope, x, d, reuse, trainable):
            with tf.name_scope(name_scope + "-conv"):
                with tf.variable_scope("conv") as scope:
                    conv = tf.contrib.layers.conv2d(
                        inputs=x, num_outputs=di,
                        kernel_size=(d, w), stride=1, padding="VALID",
                        activation_fn=tf.nn.tanh,
                        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                        weights_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_reg),
                        biases_initializer=tf.constant_initializer(1e-04),
                        reuse=reuse, trainable=trainable, scope=scope )
                    tf.get_variable_scope().reuse_variables()
                    weights = tf.get_variable('weights')
                    tf.summary.histogram('weights', weights)
                    biases = tf.get_variable('biases')
                    tf.summary.histogram('biases', biases)
                    conv_trans = tf.transpose(conv, [0, 3, 2, 1], name="conv_trans")
                    return conv_trans

        def CNN_layer(variable_scope, x1, x2, d):
            with tf.variable_scope(variable_scope):
                left_conv = convolution(name_scope="left", x=pad_for_wide_conv(x1), d=d, reuse=tf.AUTO_REUSE, trainable=True)
                right_conv = convolution(name_scope="right", x=pad_for_wide_conv(x2), d=d, reuse=tf.AUTO_REUSE, trainable=True)
                left_wp = w_pool(variable_scope="left", x=left_conv)
                left_ap = all_pool(variable_scope="left", x=left_conv)
                right_wp = w_pool(variable_scope="right", x=right_conv)
                right_ap = all_pool(variable_scope="right", x=right_conv)

                return left_wp, left_ap, right_wp, right_ap

        with tf.variable_scope('Encoder'):
            x1_expanded = tf.expand_dims(self.x1, -1)
            x2_expanded = tf.expand_dims(self.x2, -1)

            LO_0 = all_pool(variable_scope="input-left", x=x1_expanded)
            RO_0 = all_pool(variable_scope="input-right", x=x2_expanded)

            LI_1, LO_1, RI_1, RO_1 = CNN_layer(variable_scope="CNN-1", x1=x1_expanded, x2=x2_expanded, d=d0)

            sims = [cos_sim(LO_0, RO_0, axis=1), cos_sim(LO_1, RO_1, axis=1)]
            CNNs = [(LI_1, RI_1)]

            if num_layers > 1:
                for i in range(0, num_layers-1):
                    LI, LO, RI, RO = CNN_layer(variable_scope="CNN-"+str(i+2), x1=CNNs[i][0], x2=CNNs[i][1], d=di)
                    CNNs.append((LI, RI))
                    sims.append(cos_sim(LO, RO, axis=1))

            with tf.variable_scope('Cost'):
                self.cost = tf.reduce_mean(tf.square(tf.to_float(self.y1) - sims[-1]))
                self.acc = 1-self.cost
            tf.summary.scalar("cost", self.cost)
        self.prediction = CNNs[-1][0]
        print('Cost Shape {}  Preds shape {}'.format(self.cost.shape, self.prediction.shape))
        self.merged = tf.summary.merge_all()

class ABCNN_deconv():
    def __init__(self, s, w, l2_reg, d0=300, di=52, num_classes=2, num_layers=2):
        """
        Implmenentaion of ABCNNs
        (https://arxiv.org/pdf/1512.05193.pdf)
        :param s: sentence length
        :param w: filter width
        :param l2_reg: L2 regularization coefficient
        :param model_type: Type of the network(BCNN, ABCNN1, ABCNN2, ABCNN3).
        :param num_features: The number of pre-set features(not coming from CNN) used in the output layer.
        :param d0: dimensionality of word embedding(default: 300)
        :param di: The number of convolution kernels (default: 50)
        :param num_classes: The number of classes for answers.
        :param num_layers: The number of convolution layers.
        """

        self.x = tf.placeholder(tf.float32, shape=[None, di, s, 1], name="x")
        self.y = tf.placeholder(tf.float32, shape=[None, d0, s], name="y")

        # zero padding to inputs for wide convolution
        def pad_for_wide_conv(x):
            return tf.pad(x, np.array([[0, 0], [0, 0], [w - 1, w - 1], [0, 0]]), "CONSTANT", name="pad_wide_conv")

        def cos_sim(v1, v2, axis=None):
            norm1 = tf.sqrt(tf.reduce_sum(tf.square(v1), axis=axis))
            norm2 = tf.sqrt(tf.reduce_sum(tf.square(v2), axis=axis))
            dot_products = tf.reduce_sum(v1 * v2, axis=axis, name="cos_sim")

            return dot_products / (norm1 * norm2)

        def w_pool(variable_scope, x):
            with tf.variable_scope(variable_scope + "-w_pool"):
                w_ap = tf.layers.average_pooling2d(
                    inputs=x, pool_size=(1, w),
                    strides=1, padding="VALID", name="w_ap")
                return w_ap

        def all_pool(variable_scope, x):
            with tf.variable_scope(variable_scope + "-all_pool"):
                if variable_scope.startswith("input"):
                    pool_width, d = s, d0
                else: pool_width, d = s + w - 1, di
                all_ap = tf.layers.average_pooling2d(
                    inputs=x, pool_size=(1, pool_width),
                    strides=1, padding="VALID", name="all_ap")
                all_ap_reshaped = tf.reshape(all_ap, [-1, d])
                return all_ap_reshaped

        def deconvolution( x, d,reuse,trainable):
            # x = [batch, di, s, 1]
            with tf.name_scope("deconv"):
                with tf.variable_scope("deconv") as scope:
                    deconv = tf.contrib.layers.conv2d_transpose(
                    inputs= x,  num_outputs=1, kernel_size=(di,w), stride=1,
                    padding='SAME', data_format="NHWC", activation_fn=tf.nn.tanh,
                    weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                    weights_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_reg),
                    biases_initializer=tf.constant_initializer(1e-04),
                    reuse=reuse, trainable=trainable, scope=scope )
                    tf.get_variable_scope().reuse_variables()
                    weights2 = tf.get_variable('weights')
                    tf.summary.histogram('weights', weights2)
                    biases2 = tf.get_variable('biases')
                    tf.summary.histogram('biases', biases2)
                    return deconv

        def DNN_layer(variable_scope, x, d):
            # x1, x2 = [batch, d, s, 1]
            with tf.variable_scope(variable_scope):
                x_upsampled = tf.image.resize_images(x, size=(d,s), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                DI = deconvolution(x=x_upsampled, d=d, reuse=tf.AUTO_REUSE, trainable=True)
                return DI


        with tf.variable_scope("Decoder"):
            if num_layers > 1:
                DI = DNN_layer(variable_scope='DNN-1', x=self.x, d=di)
                DNNs = [DI]
                if num_layers > 2:
                    for i in range(0, num_layers-2):
                        DI = DNN_layer(variable_scope="DNN-"+str(i), x=DNNs[i], d=di)
                        DNNs.append(DI)
                DO = DNN_layer(variable_scope='DNN-'+str(num_layers), x=DNNs[-1], d=d0)
                DNNs.append(DO)
            else:
                DO = DNN_layer(variable_scope='DNN-'+str(num_layers), x=self.x, d=d0)
                DNNs.append(DO)

            with tf.variable_scope('Cost'):
                self.acc = (cos_sim(tf.squeeze(DNNs[-1], axis=3), self.y))
                self.cost = 1-self.acc
                tf.summary.scalar("cost", self.cost)
            self.prediction = tf.squeeze(DNNs[-1], axis=3)
            print('Cost Shape {}  Preds shape {}'.format(self.cost.shape, self.prediction.shape))

        self.merged = tf.summary.merge_all()
