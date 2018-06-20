import tensorflow as tf
import numpy as np


class ABDNN():
    def __init__(self, s, w, l2_reg, num_features, d0=300, di=50, num_layers=2):
        """
        :param s: sentence length
        :param w: filter width
        :param l2_reg: L2 regularization coefficient
        :param num_features: The number of pre-set features(not coming from CNN) used in the output layer.
        :param d0: dimensionality of word embedding(default: 300)
        :param di: The number of convolution kernels (default: 50)
        :param num_layers: The number of convolution layers.
        """


        self.x1 = tf.placeholder(tf.float32, shape=[None, d0, s], name="x1")
        self.y = tf.placeholder(tf.int32, shape=[None], name="y")
        self.features = tf.placeholder(tf.float32, shape=[None, num_features], name="features")

        def cos_sim(v1, v2):
            with tf.variable_scope("cos_sim"):
                norm1 = tf.sqrt(tf.reduce_sum(tf.square(v1), axis=1))
                norm2 = tf.sqrt(tf.reduce_sum(tf.square(v2), axis=1))
                dot_products = tf.reduce_sum(v1 * v2, axis=1, name="cos_sim")
                res = dot_products / (norm1 * norm2)
            return res

        def euclidean_score(v1, v2):
            euclidean = tf.sqrt(tf.reduce_sum(tf.square(v1 - v2), axis=1))
            return 1 / (1 + euclidean)

        def deconvolution(name_scope, x, d, reuse):
            with tf.name_scope(name_scope + "-deconv"):
                with tf.variable_scope("deconv") as scope:
                    deconv = tf.nn.conv2d_transpose(
                        value,
                        filter,
                        output_shape,
                        strides,
                        padding='SAME',
                        data_format='NHWC',
                        name=None
                    )

                    tf.get_variable_scope().reuse_variables()
                    weights = tf.get_variable('weights')
                    tf.summary.histogram('weights', weights)
                    biases = tf.get_variable('biases')
                    tf.summary.histogram('biases', biases)

                    return deconv

                    
        def upsample_layer(bottom,
                       n_channels, name, upscale_factor):
 
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
    
        def DNN_layer(variable_scope, x1, x2, d):
            # x1, x2 = [batch, d, s, 1]
            with tf.variable_scope(variable_scope):
                
      
                
        self.prediction = tf.contrib.layers.softmax(self.estimation)[:, 1]
        self.cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.estimation, labels=self.y), name='cost')
        #tf.add(

            #tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)),
            #name="cost")

        tf.summary.scalar("cost", self.cost)
        tf.summary.scalar("L1", tf.reduce_max(self.L1))
        if num_layers > 1:
            tf.summary.scalar("L2", tf.reduce_max(self.L2))
        self.merged = tf.summary.merge_all()

        print("=" * 50)
        print("List of Variables:")
        for v in tf.trainable_variables():
            print(v.name)
        print("=" * 50)
