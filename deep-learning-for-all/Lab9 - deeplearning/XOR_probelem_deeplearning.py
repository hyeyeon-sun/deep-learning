import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np 
 
tf.set_random_seed(777)
 
x_data = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
y_data = np.array([[0],[1],[1],[0]], dtype=np.float32)
 
X = tf.placeholder(tf.float32, [None, 2])
Y = tf.placeholder(tf.float32, [None, 1])
 
 
#in->x1,x2 두개, out-> y 하나 따라서 [2,1]
W = tf.Variable(tf.random_normal([2,1]), name = "weight")
#b의 형태는 언제나 out의 형태와 같음 -> [1]
b = tf.Variable(tf.random_normal([1]), name = "bias")
hypothesis = tf.sigmoid(tf.matmul(X,W)+b)
 
# local minimum에 빠지지 않도록 !
cost = tf.reduce_mean(Y*tf.log(hypothesis)+(1-Y)*tf.log(1-hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
 
#hypothesys>0.5 true(1) else false(0) ->cast함수가 함
predicted = tf.cast(hypothesis>0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,Y), dtype = tf.float32))
 
with tf.Session() as sess:
    #Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())
 
    for step in range(10001):
        _, cost_val, w_val = sess.run(
                    [train, cost, W], feed_dict = {X: x_data, Y: y_data}
 
        )
        if step %100 ==0:
            print(step, cost_val, w_val)
 
    #Accuracy report
    h, c, a = sess.run(
                   [hypothesis, predicted, accuracy], feed_dict={X:x_data, Y:y_data} 
    )
    print("\nHypothesis: ", h, "\nCorrect: ", c, "\nAccuracy: ", a)