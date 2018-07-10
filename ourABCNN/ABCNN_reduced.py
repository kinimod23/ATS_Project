import tensorflow as tf
import numpy as np


class ABCNN():
    def __init__(self, s, w, l2_reg, model_type, num_features, d0=300, di=52, num_classes=2, num_layers=2):
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

        self.x1 = tf.placeholder(tf.float32, shape=[None, d0, s], name="x1")
        self.x2 = tf.placeholder(tf.float32, shape=[None, d0, s], name="x2")
        if model_type == 'convolution':
            self.y = tf.placeholder(tf.int32, shape=[None], name="y")
        else:
            self.y = tf.placeholder(tf.float32, shape=[None, d0, s], name="y")
        self.features = tf.placeholder(tf.float32, shape=[None, num_features], name="features")

        # zero padding to inputs for wide convolution
        def pad_for_wide_conv(x):
            return tf.pad(x, np.array([[0, 0], [0, 0], [w - 1, w - 1], [0, 0]]), "CONSTANT", name="pad_wide_conv")

        def euclidean_score(v1, v2):
            euclidean = tf.sqrt(tf.reduce_sum(tf.square(v1 - v2)))
            return 1 / (1 + euclidean)

        def cos_sim(v1, v2):
            norm1 = tf.sqrt(tf.reduce_sum(tf.square(v1), axis=1))
            norm2 = tf.sqrt(tf.reduce_sum(tf.square(v2), axis=1))
            dot_products = tf.reduce_sum(v1 * v2, axis=1, name="cos_sim")

            return dot_products / (norm1 * norm2)

        def cos_sim2(v1, v2):
            norm1 = tf.sqrt(tf.reduce_sum(tf.square(v1)))
            norm2 = tf.sqrt(tf.reduce_sum(tf.square(v2)))
            dot_products = tf.reduce_sum(v1 * v2, name="cos_sim2")

            return dot_products / (norm1 * norm2)

        def w_pool(variable_scope, x):
            # x: [batch, di, s+w-1, 1]
            # attention: [batch, s+w-1]
            with tf.variable_scope(variable_scope + "-w_pool"):
                w_ap = tf.layers.average_pooling2d(
                    inputs=x,
                    # (pool_height, pool_width)
                    pool_size=(1, w),
                    strides=1,
                    padding="VALID",
                    name="w_ap"
                )
                # [batch, di, s, 1]
                return w_ap

        def all_pool(variable_scope, x):
            with tf.variable_scope(variable_scope + "-all_pool"):
                if variable_scope.startswith("input"):
                    pool_width = s
                    d = d0
                else:
                    pool_width = s + w - 1
                    d = di

                all_ap = tf.layers.average_pooling2d(
                    inputs=x,
                    # (pool_height, pool_width)
                    pool_size=(1, pool_width),
                    strides=1,
                    padding="VALID",
                    name="all_ap"
                )
                # [batch, di, 1, 1]

                # [batch, di]
                all_ap_reshaped = tf.reshape(all_ap, [-1, d])
                #all_ap_reshaped = tf.squeeze(all_ap, [2, 3])

                return all_ap_reshaped

        def convolution(name_scope, x, d, reuse, trainable):
            with tf.name_scope(name_scope + "-conv"):
                with tf.variable_scope("conv") as scope:
                    conv = tf.contrib.layers.conv2d(
                        inputs=x,
                        num_outputs=di,
                        kernel_size=(d, w),
                        stride=1,
                        padding="VALID",
                        activation_fn=tf.nn.tanh,
                        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                        weights_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_reg),
                        biases_initializer=tf.constant_initializer(1e-04),
                        reuse=reuse,
                        trainable=trainable,
                        scope=scope
                    )
                    tf.get_variable_scope().reuse_variables()
                    weights = tf.get_variable('weights')
                    tf.summary.histogram('weights', weights)
                    biases = tf.get_variable('biases')
                    tf.summary.histogram('biases', biases)
                    # Weight: [filter_height, filter_width, in_channels, out_channels]
                    # output: [batch, 1, input_width+filter_Width-1, out_channels] == [batch, 1, s+w-1, di]

                    # [batch, di, s+w-1, 1]
                    conv_trans = tf.transpose(conv, [0, 3, 2, 1], name="conv_trans")
                    return conv_trans

        def CNN_layer(variable_scope, x1, x2, d):
            # x1, x2 = [batch, d, s, 1]
            with tf.variable_scope(variable_scope):

                if model_type == 'convolution' or model_type == 'End2End':
                    left_conv = convolution(name_scope="left", x=pad_for_wide_conv(x1), d=d, reuse=tf.AUTO_REUSE, trainable=True)
                    right_conv = convolution(name_scope="right", x=pad_for_wide_conv(x2), d=d, reuse=tf.AUTO_REUSE, trainable=True)
                else:
                    left_conv = convolution(name_scope="left", x=pad_for_wide_conv(x1), d=d, reuse=tf.AUTO_REUSE, trainable=False)
                    right_conv = convolution(name_scope="right", x=pad_for_wide_conv(x2), d=d, reuse=tf.AUTO_REUSE, trainable=False)

                left_wp = w_pool(variable_scope="left", x=left_conv)
                left_ap = all_pool(variable_scope="left", x=left_conv)
                right_wp = w_pool(variable_scope="right", x=right_conv)
                right_ap = all_pool(variable_scope="right", x=right_conv)

                return left_wp, left_ap, right_wp, right_ap

        def deconvolution( x, d,reuse,trainable):
            # x = [batch, di, s, 1]
            with tf.name_scope("deconv"):
                with tf.variable_scope("deconv") as scope:
                    deconv = tf.contrib.layers.conv2d_transpose(
                    inputs= x, #[batch, height, width, in_channels]
                    num_outputs=1,
                    kernel_size=(di,w),
                    stride=1,
                    padding='SAME',
                    data_format="NHWC",
                    activation_fn=tf.nn.tanh,
                    weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                    weights_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_reg),
                    biases_initializer=tf.constant_initializer(1e-04),
                    reuse=reuse,
                    trainable=trainable,
                    scope=scope
                    )

                    tf.get_variable_scope().reuse_variables()
                    weights2 = tf.get_variable('weights')
                    tf.summary.histogram('weights', weights2)
                    biases2 = tf.get_variable('biases')
                    tf.summary.histogram('biases', biases2)

                    return deconv

        def upsample_layer(bottom, n_channels, name, upscale_factor):
            kernel_size = 2*upscale_factor - upscale_factor%2
            stride = upscale_factor
            strides = [1, stride, stride, 1]
            with tf.variable_scope(name):
                # Shape of the bottom tensor
                in_shape = tf.shape(bottom)

                h = ((in_shape[1] - 1) * stride) + 1
                w = ((in_shape[2] - 1) * stride) + 1
                new_shape = [in_shape[0], h, w, n_channels]
                output_shape = tf.stack(new_shape)

                filter_shape = [kernel_size, kernel_size, n_channels, n_channels]

                weights = get_bilinear_filter(filter_shape,upscale_factor)
                deconv = tf.nn.conv2d_transpose(bottom, weights, output_shape,
                                                strides=strides, padding='SAME')
            return deconv

        def get_bilinear_filter(filter_shape, upscale_factor):
            ##filter_shape is [width, height, num_in_channels, num_out_channels]
            kernel_size = filter_shape[1]
            ### Centre location of the filter for which value is calculated
            if kernel_size % 2 == 1:
             centre_location = upscale_factor - 1
            else:
              centre_location = upscale_factor - 0.5

            bilinear = np.zeros([filter_shape[0], filter_shape[1]])
            for x in range(filter_shape[0]):
                for y in range(filter_shape[1]):
                   ##Interpolation Calculation
                   value = (1 - abs((x - centre_location)/ upscale_factor)) * (1 - abs((y - centre_location)/ upscale_factor))
                   bilinear[x, y] = value
            weights = np.zeros(filter_shape)
            for i in range(filter_shape[2]):
                weights[:, :, i, i] = bilinear
            init = tf.constant_initializer(value=weights,
                                            dtype=tf.float32)

            bilinear_weights = tf.get_variable(name="decon_bilinear_filter", initializer=init,
                                 shape=weights.shape)
            return bilinear_weights

        def DNN_layer(variable_scope, x, d):
            # x1, x2 = [batch, d, s, 1]
            with tf.variable_scope(variable_scope):
                x_upsampled = tf.image.resize_images(x, size=(d,s), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                DI = deconvolution(x=x_upsampled, d=d, reuse=tf.AUTO_REUSE, trainable=True)
                return DI

        with tf.variable_scope("Encoder"):
            x1_expanded = tf.expand_dims(self.x1, -1)
            x2_expanded = tf.expand_dims(self.x2, -1)

            LO_0 = all_pool(variable_scope="input-left", x=x1_expanded)
            RO_0 = all_pool(variable_scope="input-right", x=x2_expanded)

            LI_1, LO_1, RI_1, RO_1 = CNN_layer(variable_scope="CNN-1", x1=x1_expanded, x2=x2_expanded, d=d0)

            sims = [cos_sim(LO_0, RO_0), cos_sim(LO_1, RO_1)]
            CNNs = [(LI_1, RI_1)]

            if num_layers > 1:
                for i in range(0, num_layers-1):
                    LI, LO, RI, RO = CNN_layer(variable_scope="CNN-"+str(i+2), x1=CNNs[i][0], x2=CNNs[i][1], d=di)
                    CNNs.append((LI, RI))
                    sims.append(cos_sim(LO, RO))

            if model_type == 'convolution':
                with tf.variable_scope("output-layer"):
                    self.output_features = tf.concat([self.features, tf.stack(sims, axis=1)], axis=1, name="output_features")

                    self.estimation = tf.contrib.layers.fully_connected(
                        inputs=self.output_features,
                        num_outputs=num_classes,
                        activation_fn=None,
                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                        weights_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_reg),
                        biases_initializer=tf.constant_initializer(1e-04),
                        scope="FC"
                    )

                self.prediction = tf.contrib.layers.softmax(self.estimation)[:, 1]
                with tf.variable_scope('Cost'):

                    self.cost = tf.add(
                    tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.estimation, labels=self.y)),
                    tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)),
                    name="cost")
                    self.cost2 = self.cost

                tf.summary.scalar("cost", self.cost)
                tf.summary.scalar("cost2", self.cost2)

        if model_type != 'convolution':
            with tf.variable_scope("Decoder"):
                if num_layers > 1:
                    DI = DNN_layer(variable_scope='DNN-1', x=CNNs[-1][0], d=di)
                    DNNs = [DI]

                    if num_layers > 2:
                        for i in range(0, num_layers-2):
                            DI = DNN_layer(variable_scope="DNN-"+str(i), x=DNNs[i], d=di)
                            DNNs.append(DI)

                    DO = DNN_layer(variable_scope='DNN-'+str(num_layers), x=DNNs[-1], d=d0)
                    DNNs.append(DO)
                else:
                    DO = DNN_layer(variable_scope='DNN-'+str(num_layers), x=CNNs[-1][0], d=d0)
                    DNNs.append(DO)

            with tf.variable_scope('Cost'):
                self.cost2 = 1/euclidean_score(tf.squeeze(DNNs[-1], axis=3), self.y)
                self.cost = 1/(cos_sim2(tf.squeeze(DNNs[-1], axis=3), self.y))
                tf.summary.scalar("cost", self.cost)
                tf.summary.scalar("cost2", self.cost2)
            self.output_features = self.features

            self.prediction = DNNs[-1]

        self.merged = tf.summary.merge_all()

        print("=" * 50)
        print("List of Variables:")
        for v in tf.trainable_variables():
            print(v.name, v.shape)
        print("=" * 50)
