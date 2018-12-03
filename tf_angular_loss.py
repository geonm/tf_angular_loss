import os
import sys
import numpy as np
import tensorflow as tf


def get_angular_loss(input_labels, anchor_features, pos_features, degree=45, batch_size=10, with_l2reg=False):
    '''
    #NOTE: degree is degree!!! not radian value
    '''
    if with_l2reg:
        reg_anchor = tf.reduce_mean(tf.reduce_sum(tf.square(anchor_features), 1))
        reg_positive = tf.reduce_mean(tf.reduce_sum(tf.square(pos_features), 1))
        l2loss = tf.multiply(0.25 * 0.002, reg_anchor + reg_positive, name='l2loss_angular')
    else:
        l2loss = 0.0

    alpha = np.deg2rad(degree)
    sq_tan_alpha = np.tan(alpha) ** 2

    #anchor_features = tf.nn.l2_normalize(anchor_features)
    #pos_features = tf.nn.l2_normalize(pos_features)
    
    # 2(1+(tan(alpha))^2 * xaTxp)
    #batch_size = 10
    xaTxp = tf.matmul(anchor_features, pos_features, transpose_a=False, transpose_b=True)
    sim_matrix_1 = tf.multiply(2.0 * (1.0 + sq_tan_alpha) * xaTxp, tf.eye(batch_size, dtype=tf.float32))

    # 4((tan(alpha))^2(xa + xp)Txn
    xaPxpTxn = tf.matmul((anchor_features + pos_features), pos_features, transpose_a=False, transpose_b=True)
    sim_matrix_2 = tf.multiply(4.0 * sq_tan_alpha * xaPxpTxn, tf.ones_like(xaPxpTxn, dtype=tf.float32) - tf.eye(batch_size, dtype=tf.float32))

    # similarity_matrix
    similarity_matrix = sim_matrix_1 + sim_matrix_2

    # do softmax cross-entropy
    lshape = tf.shape(input_labels)
    #assert lshape.shape == 1
    labels = tf.reshape(input_labels, [lshape[0], 1])

    labels_remapped = tf.to_float(tf.equal(labels, tf.transpose(labels)))
    labels_remapped /= tf.reduce_sum(labels_remapped, 1, keepdims=True)

    xent_loss = tf.nn.softmax_cross_entropy_with_logits(logits=similarity_matrix, labels=labels_remapped)
    xent_loss = tf.reduce_mean(xent_loss, name='xentropy_angular')

    return l2loss + xent_loss

if __name__ == '__main__':
    ## test
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    dims_vector = 256
    np_anchor = np.random.rand(10, dims_vector)#np.array([[1,2,3,4,5],[6,7,8,9,10]])
    np_pos = np.random.rand(10, dims_vector)#np.array([[1.5, 2.5, 3.5, 4.5, 5.5], [6.5, 7.5, 8.5, 9.5, 10.5]])
    np_pos_same = np_anchor.copy()

    anchor = tf.convert_to_tensor(np_anchor, dtype=tf.float32)
    pos = tf.convert_to_tensor(np_pos, dtype=tf.float32)
    pos_same = tf.convert_to_tensor(np_pos_same, dtype=tf.float32)

    real_labels = tf.convert_to_tensor(np.arange(start=0, stop=10, dtype='int32'))

    degree = 30

    angular_loss = get_angular_loss(real_labels, anchor, pos, degree=degree)
    angular_loss_same = get_angular_loss(real_labels, anchor, pos_same, degree=degree)

    npairs_loss = tf.contrib.losses.metric_learning.npairs_loss(real_labels, anchor, pos)
    npairs_loss_same = tf.contrib.losses.metric_learning.npairs_loss(real_labels, anchor, pos_same)
    ##### show outputs
    sess = tf.Session()

    #print(sess.run(labels), np_anchor.shape)

    out_angular_loss, out_angular_loss_same, out_npairs_loss, out_npairs_loss_same = sess.run([angular_loss, angular_loss_same, npairs_loss, npairs_loss_same])

    print('\nangular_loss: \n', out_angular_loss, out_angular_loss_same)
    print('\nnpairs_loss: \n', out_npairs_loss, out_npairs_loss_same)
    print('')
