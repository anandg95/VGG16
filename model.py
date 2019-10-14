import tensorflow as tf


def conv_layer(x, n_filters, kernel_size, name):
    with tf.variable_scope(name) as scope:
        y = tf.layers.conv2d(
            x,
            filters=n_filters,
            kernel_size=kernel_size,
            padding="same",
            kernel_initializer=tf.keras.initializers.TruncatedNormal(
                mean=0, stddev=0.001
            ),
            bias_initializer=tf.keras.initializers.TruncatedNormal(
                mean=0, stddev=0.001
            ),
        )
        return y


def fc_layer(x, output_size, name):
    with tf.variable_scope(name) as scope:
        y = tf.contrib.layers.fully_connected(x, num_outputs=output_size)
        return y


def max_pool(x, name, pool_size=[2, 2], strides=2):
    with tf.variable_scope(name) as scope:
        y = tf.layers.max_pooling2d(
            x, pool_size=pool_size, strides=strides, padding="same"
        )
        return y


def dropout(x, rate=0.5, is_training=True):
    return tf.contrib.layers.dropout(x, keep_prob=rate, is_training=is_training)


def model(x, num_classes=1000, is_training=True):
    conv1_1 = conv_layer(x, 64, [3, 3], "conv1_1")
    conv1_2 = conv_layer(conv1_1, 64, [3, 3], "conv1_2")
    pool1 = max_pool(conv1_2, "pool1")
    # =======================================================
    conv2_1 = conv_layer(pool1, 128, [3, 3], "conv2_1")
    conv2_2 = conv_layer(conv2_1, 128, [3, 3], "conv2_2")
    pool2 = max_pool(conv2_2, "pool2")
    # =======================================================
    conv3_1 = conv_layer(pool2, 256, [3, 3], "conv3_1")
    conv3_2 = conv_layer(conv3_1, 256, [3, 3], "conv3_2")
    conv3_3 = conv_layer(conv3_2, 256, [1, 1], "conv3_3")
    pool3 = max_pool(conv3_3, "pool3")
    # =======================================================
    conv4_1 = conv_layer(pool3, 512, [3, 3], "conv4_1")
    conv4_2 = conv_layer(conv4_1, 512, [3, 3], "conv4_2")
    conv4_3 = conv_layer(conv4_2, 512, [1, 1], "conv4_3")
    pool4 = max_pool(conv4_3, "pool4")
    # =======================================================
    conv5_1 = conv_layer(pool4, 512, [3, 3], "conv5_1")
    conv5_2 = conv_layer(conv5_1, 512, [3, 3], "conv5_2")
    conv5_3 = conv_layer(conv5_2, 512, [1, 1], "conv5_3")
    pool5 = max_pool(conv5_3, "pool5")

    # ========================================================
    flat = tf.contrib.layers.flatten(pool5)
    fc1 = fc_layer(flat, 4096, "fc1")
    dropout1 = dropout(fc1, is_training)
    fc2 = fc_layer(dropout1, 4096, "fc2")
    dropout2 = dropout(fc2, is_training)
    fc3 = fc_layer(dropout2, num_classes, "fc3")
    # =======================================================

    # pred_probs = tf.contrib.layers.softmax(fc3, name="softmax")
    return fc3


def loss_op(logits, labels):
    loss_op = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits),
        name="cost",
    )
    # tf.summary.scalar("loss", loss_op)
    return loss_op


def optimizer_op(learning_rate):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name="Adam")
    return optimizer


def accuracy_op(logits, sparse_label):
    correct_pred = tf.math.equal(tf.math.argmax(logits, 1), sparse_label)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name="accuracy")
    # tf.summary.scalar("accuracy", accuracy)
    return accuracy


def pred_probs(logits):
    return tf.nn.softmax(logits=logits, name="softmax")
