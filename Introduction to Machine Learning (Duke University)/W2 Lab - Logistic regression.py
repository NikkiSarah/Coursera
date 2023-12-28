import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
import tensorflow as tf
from tqdm import trange

# check the backend and change if required
import matplotlib as mpl

mpl_backend = mpl.get_backend()
if mpl_backend != "Qt5Agg":
    mpl.use("Qt5Agg")
else:
    pass

# import the mnist dataset
(train, train_labels), (test, test_labels) = mnist.load_data()

# Dataset statistics
print('Training image data: {0}'.format(train.shape))
print('Testing image data: {0}'.format(test.shape))
print('28 x 28 = {0}'.format(train.shape[1] * train.shape[2]))

print('\nTrain labels: {0}'.format(train_labels.shape))
counts = np.bincount(train_labels)
labels = np.arange(len(counts))
print('\nLabel distribution: {0}'.format(list(zip(labels, counts))))

plt.bar(labels, counts)

# example image
idx = 0
plt.imshow(train[idx], cmap='grey')
plt.title(f"Label: {train_labels[idx]}")

# scale images to the [0, 1] range
train = train.astype("float32") / 255
test = test.astype("float32") / 255

# flatten the images so they have shape (num_examples, shape[0]*shape[1])
train = train.reshape((train.shape[0], -1))
test = test.reshape((test.shape[0], -1))
print('Training image data: {0}'.format(train.shape))
print('Testing image data: {0}'.format(test.shape))

# convert the labels to a OHE representation
num_classes = len(labels)
train_labels = keras.utils.to_categorical(train_labels, num_classes)
test_labels = keras.utils.to_categorical(test_labels, num_classes)

# create the model
# define the input placeholder
x = tf.keras.Input(shape=(train.shape[1],), dtype=tf.float32)
# define the linear transformation
W = tf.Variable(tf.zeros((train.shape[1], num_classes), dtype=tf.float32))
b = tf.Variable(tf.zeros((num_classes,), dtype=tf.float32))
y = tf.matmul(x, W) + b

# define the loss and optimiser
y_ = tf.keras.Input(shape=(num_classes, ), dtype=tf.float32)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
optimiser = tf.optimizers.SGD()

# train the model
for epoch in trange(1000):
    idxs = np.random.choice(train.shape[0], 100, replace=False)
    image_batch, label_batch = train[idxs], train_labels[idxs]

    with tf.GradientTape() as tape:
        train_pred = tf.matmul(image_batch, W) + b
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label_batch, logits=train_pred))
    gradients = tape.gradient(loss, [W, b])
    optimiser.apply_gradients(zip(gradients, [W, b]))


# test the trained model
def test_predictions(data, labels):
    preds = tf.matmul(data, W) + b
    pred_probs = tf.nn.softmax(preds)
    pred_labels = np.argmax(pred_probs, axis=1)
    true_labels = np.argmax(labels, axis=1)
    accuracy = np.mean(pred_labels == true_labels)
    print("Accuracy: {:.2%}".format(accuracy))


test_predictions(train, train_labels)
test_predictions(test, test_labels)
