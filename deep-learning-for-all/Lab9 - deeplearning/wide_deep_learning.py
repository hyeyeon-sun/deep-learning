import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np 
 
tf.set_random_seed(777)
 
x_data = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
y_data = np.array([[0],[1],[1],[0]], dtype=np.float32)
 
X = tf.placeholder(tf.float32, [None, 2])
Y = tf.placeholder(tf.float32, [None, 1])
 
#//레이어 없이//#
#in->x1,x2 두개, out-> y 하나 따라서 [2,1]
#W = tf.Variable(tf.random_normal([2,1]), name = "weight")
#b의 형태는 언제나 out의 형태와 같음 -> [1]
#b = tf.Variable(tf.random_normal([1]), name = "bias")
#hypothesis = tf.sigmoid(tf.matmul(X,W)+b)
#-> 작동하지 않는다.
 
 
#//층하나를 만듦//#
#지난시간에 두개의 유닛을 행렬을 통해 하나로 합친다고 했었죠?
#unit -> w shape([1,2]), b shape([1]) 둘이 합쳐져서 밑과 같습니다.
#어려우면 그냥 in이 2이고, 두개의 유닛이기 때문에 out도 2이다라고 이해해도 괜찮습니다.
#W1 = tf.Variable(tf.random_normal([2,2]), name = 'weight1')
#b1 = tf.Variable(tf.random_normal([2]), name = 'bias1')
#layer1 = tf.sigmoid(tf.matmul(X,W1)+b1)
#W2 = tf.Variable(tf.random_normal([2,1]), name = 'weight2')
#b2 = tf.Variable(tf.random_normal([1]), name = 'bias2')
#hypothesis = tf.sigmoid(tf.matmul(layer1,W2)+b2)
 
 
#//wide model//#
#하나의 층에서 unit을 여러개로 만드는 것 = 출력을 여러개로 만드는 것
#W1 = tf.Variable(tf.random_normal([2,10]), name = 'weight1')
#b1 = tf.Variable(tf.random_normal([10]), name = 'bias1')
#layer1 = tf.sigmoid(tf.matmul(X,W1)+b1)
#W2 = tf.Variable(tf.random_normal([10,1]), name = 'weight2')
#b2 = tf.Variable(tf.random_normal([1]), name = 'bias2')
#hypothesis = tf.sigmoid(tf.matmul(layer1,W2)+b2)
 
 
#//wide&deep model//#
W1 = tf.Variable(tf.random_normal([2, 10]), name='weight1')
b1 = tf.Variable(tf.random_normal([10]), name='bias1')
layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)
 
W2 = tf.Variable(tf.random_normal([10, 10]), name='weight2')
b2 = tf.Variable(tf.random_normal([10]), name='bias2')
layer2 = tf.sigmoid(tf.matmul(layer1, W2) + b2)
 
W3 = tf.Variable(tf.random_normal([10, 10]), name='weight3')
b3 = tf.Variable(tf.random_normal([10]), name='bias3')
layer3 = tf.sigmoid(tf.matmul(layer2, W3) + b3)
 
W4 = tf.Variable(tf.random_normal([10, 1]), name='weight4')
b4 = tf.Variable(tf.random_normal([1]), name='bias4')
hypothesis = tf.sigmoid(tf.matmul(layer3, W4) + b4)
 
 
 
# local minimum에 빠지지 않도록 !
cost = -tf.reduce_mean(Y*tf.log(hypothesis)+(1-Y)*tf.log(1-hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
 
#hypothesys>0.5 true(1) else false(0) ->cast함수가 함
predicted = tf.cast(hypothesis>0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,Y), dtype = tf.float32))
 
with tf.Session() as sess:
    #Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())
 
    for step in range(10001):
        _, cost_val, = sess.run([train, cost], feed_dict = {X: x_data, Y: y_data})
        if step %100 ==0:
            print(step, cost_val)
 
    #Accuracy report
    h, p, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X:x_data, Y:y_data} )
    print("\nHypothesis: ", h, "\nCorrect: ", p, "\nAccuracy: ", a)
