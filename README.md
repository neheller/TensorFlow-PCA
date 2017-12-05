# TensorFlow-PCA
An implementation of principle component analysis using TensorFlow's singular value decomposition

## Gist
We were talking about the SVD in my matrix theory class and how it can be used for principle compnent analysis, so I thought it would be instructive to implement this using TensorFlow's [tf.svd function](https://www.tensorflow.org/api_docs/python/tf/svd) on the [MNIST dataset](http://yann.lecun.com/exdb/mnist/).

## Files
* **pca.py**: The main (and only) script. Run with `$ python pca.py <dims> <index>`
  * _dims_: The number of dimensions to reduce to (from 784)
  * _index_: The index of the image to show at the end
* **testX.npy**: The MNIST testing data saved as a numpy array
* **testY.npy**: The MNIST testing labels saved as a numpy array
* **trainX.npy**: (Too big for GitHub)[http://distrob.cs.umn.edu/trainX.npy]
* **trainY.npy**: The MNIST training labels saved as a numpy array

## Behavior
Will compute lower-dimensional representations of the input images, then attempt to reconstruct them. Shows a single image before and after reconstruction.

![alt text](https://github.com/NewtonsLadle/TensorFlow-PCA/master/original.png "The original image at index 5 in the training set")

![alt text](https://github.com/NewtonsLadle/TensorFlow-PCA/master/reconstructed.png "The reconstruced image using 50 principle components")
