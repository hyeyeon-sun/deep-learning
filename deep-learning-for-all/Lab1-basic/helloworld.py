import tensorflow.compat.v1 as try:
    tf.disable_v2_behavior()

#<tensorflow hello world>
#"Hello, TensorFlow!"라는 문자열이 들어 있는 하나의 노드를 생성하는 것
hello = tf.constant("Hello, TensorFlow!")
#session - computational graph를 실행하기 위해 만들어야함
sess = tf.Session()
#hello라는 노드 생성
print(sess.run(hello))
