import tensorflow as tf

# add a constant operation to the default graph
hello = tf.constant("Hello, Tensorflow!")
# add a second constant operation to the graph
a = tf.constant(3., dtype=tf.float32)
# add a third constant operation to the graph
b = tf.constant(4., dtype=tf.float32)
# create a tensorflow operation that adds tensors a and b, and produces a new tensor
sum_ab = tf.add(a, b)

print("The first constant tensor has value: {}".format(a))
print("The result of the add operation has value: {}".format(sum_ab))

x = tf.constant([[1.], [2.]], dtype=tf.float32)
W = tf.constant([[3., 4.], [5., 6.]], dtype=tf.float32)
# perform matrix-vector multiplication W*x
y = tf.matmul(W, x)

# define the input data
x = tf.constant([[1.], [2.]], dtype=tf.float32)
# define the constant weight matrix W
W = tf.constant([[3., 4.], [5., 6.]], dtype=tf.float32)
# perform matrix multiplication
y = tf.matmul(W, x)

x = tf.constant([[2.], [1.]], dtype=tf.float32)
init_val = tf.random.normal(shape=(2, 2))
W = tf.Variable(init_val)
y = tf.matmul(W, x)