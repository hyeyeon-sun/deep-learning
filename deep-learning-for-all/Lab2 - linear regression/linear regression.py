import tensoflow.compat.v1 as tf
tf.disable_v2_behavior()

x_train = [1,2,3]
y_train = [1,2,3]

# Variable : tensorflow가 사용하는 variable
# tensorflow가 값을 임의로 변경할 수 있다 -> train 을 통해 값을 알아서 갱신해야 하므로
# 만드는 법 : shape와 name을 준다.
w = tf.Variable(tf.random_normal([1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')
# our hypothesis Wx+b
hypothesis = x_train*w +b

# cost function
cost = tf.reduce_mean(tf.square(hypothesis-y_train))

#optimizer라는 노드를 불러오고,
#optimizer의 minimize 함수를 이용해 cost를 최소화한다.
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
train = optimizer.minimize(cost)

#그래프를 세션에서 실행
sess = tf.Session()
# 광역변수 w,b를 초기화한다.
sess.run(tf.global_variables_initializer())

#Fit the line
for step in range(2001):
    sess.run(train)
    if step%20 ==0:
        print(step, sess.run(cost), sess.run(w), sess.run(b))
