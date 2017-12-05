import numpy as np
import tensorflow as tf
import sys

import matplotlib.pyplot as plt

def normalize(X):
    tmp = np.divide(X, 255.0)
    return tmp - np.mean(tmp)

def denormalize(Rn):
    tmp = np.subtract(Rn, np.min(Rn))
    tmp2 = np.multiply(np.divide(tmp, np.max(tmp)), 255).astype(np.uint8)
    return tmp2

def showim(lin):
    twodarr = np.array(lin).reshape((28,28))
    plt.imshow(twodarr, cmap="gray")
    plt.show()



if __name__ == '__main__':

    dims = 50
    index = 0

    if (len(sys.argv) > 1):
        dims = int(sys.argv[1])
    if (len(sys.argv) > 2):
        index = int(sys.argv[2])

    X = np.load('testX.npy')
    Xn = normalize(X)

    Xtf = tf.placeholder(tf.float32, shape=[X.shape[0], X.shape[1]])

    stf, utf, vtf = tf.svd(Xtf)

    tvtf = tf.slice(vtf, [0, 0], [784, dims])

    Ttf = tf.matmul(Xtf, tvtf)
    Rtf = tf.matmul(Ttf, tvtf, transpose_b=True)

    with tf.Session() as sess:
        Rn = sess.run(Rtf, feed_dict = {
            Xtf: Xn
        })

    R = denormalize(Rn)

    showim(X[index])
    showim(R[index])
