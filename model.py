import numpy as np
import tensorflow as tf


def compute_pure_ratio(ind1, ind2, indices, noise_or_not):
    num_remember = len(ind1)
    pure_ratio_1 = np.sum(noise_or_not[indices[ind1]]) / float(num_remember)
    pure_ratio_2 = np.sum(noise_or_not[indices[ind2]]) / float(num_remember)
    return pure_ratio_1, pure_ratio_2

    
class ConvLayer(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, padding):
        super().__init__()
        self.conv = tf.keras.layers.Conv2D(filters, kernel_size, strides, padding, use_bias=False)
        self.bn = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.relu = tf.keras.layers.LeakyReLU(alpha=0.01)

    def call(self, inputs, training):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        return self.relu(x)

    
class CNNModel(tf.keras.Model):
    def __init__(self, n_outputs, drop_rate, top_bn=False):
        super().__init__()
        self.drop_rate = drop_rate  
        self.top_bn = top_bn  
        
        self.conv1 = ConvLayer(128, 3, 1, "same")
        self.conv2 = ConvLayer(128, 3, 1, "same")
        self.conv3 = ConvLayer(128, 3, 1, "same")
        self.pool3 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)

        self.conv4 = ConvLayer(256, 3, 1, "same")
        self.conv5 = ConvLayer(256, 3, 1, "same")
        self.conv6 = ConvLayer(256, 3, 1, "same")
        self.pool6 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)

        self.conv7 = ConvLayer(512, 3, 1, "valid")
        self.conv8 = ConvLayer(256, 3, 1, "valid")
        self.conv9 = ConvLayer(128, 3, 1, "valid")

        self.flatten = tf.keras.layers.Flatten()
        self.fc = tf.keras.layers.Dense(n_outputs)

    def call(self, inputs, training=False):
        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        x = self.pool3(x)

        if training:  
            noise_shape = [tf.shape(x)[0].numpy(), tf.shape(x)[1].numpy(), tf.shape(x)[2].numpy(), 1]  # 動的な形状
            x = tf.nn.dropout(x, rate=self.drop_rate, noise_shape=noise_shape)

        x = self.conv4(x, training=training)
        x = self.conv5(x, training=training)
        x = self.conv6(x, training=training)
        x = self.pool6(x)

        if training: 
            noise_shape = [tf.shape(x)[0].numpy(), tf.shape(x)[1].numpy(), tf.shape(x)[2].numpy(), 1]  # 動的な形状
            x = tf.nn.dropout(x, rate=self.drop_rate, noise_shape=noise_shape)

        x = self.conv7(x, training=training)
        x = self.conv8(x, training=training)
        x = self.conv9(x, training=training)

        pool_size=tf.shape(x)[1].numpy()
        strides=tf.shape(x)[1].numpy()
        x = tf.nn.avg_pool(x, ksize=[1, pool_size, pool_size, 1], strides=[1, strides, strides, 1], padding="VALID")

        h = self.flatten(x)
        h = self.fc(h)
        if self.top_bn:
            h = tf.layers.batch_normalization(h, momentum=0.9, epsilon=1e-5, training=training)
        logits = h
        predicts = tf.argmax(tf.nn.softmax(h, axis=-1), axis=-1)
        return logits, predicts
    
    

def coteach_loss(logits1, logits2, labels, forget_rate):
    # compute loss
    raw_loss1 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits1, labels=labels)
    raw_loss2 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits2, labels=labels)

    # sort and get low loss indices
    ind1_sorted = tf.argsort(raw_loss1, axis=-1, direction="ASCENDING", stable=True)
    ind2_sorted = tf.argsort(raw_loss2, axis=-1, direction="ASCENDING", stable=True)
    num_remember = tf.cast((1.0 - forget_rate) * ind1_sorted.shape[0], dtype=tf.int32)
    ind1_update = ind1_sorted[:num_remember]
    ind2_update = ind2_sorted[:num_remember]

    # update logits and compute loss again
    logits1_update = tf.gather(logits1, ind2_update, axis=0)
    labels1_update = tf.gather(labels, ind2_update, axis=0)
    loss1 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits1_update, labels=labels1_update)
    loss1 = tf.reduce_sum(loss1) / tf.cast(num_remember, dtype=tf.float32)

    logits2_update = tf.gather(logits2, ind1_update, axis=0)
    labels2_update = tf.gather(labels, ind1_update, axis=0)
    loss2 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits2_update, labels=labels2_update)
    loss2 = tf.reduce_sum(loss2) / tf.cast(num_remember, dtype=tf.float32)

    return loss1, loss2, ind1_update, ind2_update


class CoTeachingModel(tf.keras.Model):
    def __init__(self, n_outputs, batch_size=128, drop_rate=0.25, top_bn=False):
        super().__init__()
        
        self.n_outputs = n_outputs
        self.batch_size = batch_size
        self.drop_rate = drop_rate
        self.top_bn = top_bn
        
        self.cnn_model_1 = CNNModel(self.n_outputs, self.drop_rate, self.top_bn)
        self.cnn_model_2 = CNNModel(self.n_outputs, self.drop_rate, self.top_bn)

    def call(self, inputs, training=False):
        logits1, self.predicts1 = self.cnn_model_1(inputs, training=training)
        logits2, self.predicts2 = self.cnn_model_2(inputs, training=training)
        return logits1, self.predicts1, logits2, self.predicts2

    
