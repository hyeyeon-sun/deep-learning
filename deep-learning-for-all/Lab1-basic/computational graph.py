import tensoflow.compat.v1 as tf
tf.disable_v2_behavior()

node1 = tf.constant(3.0)
node2 = tf.constant(4.0)
node3 = tf.add(node1,node2)

#print("node1:",node1, "node2: ", node2)
#print("node3: ", node3)
#텐서의 데이터 타입만 나온다.

sess = tf.Session()
print("sess.run(node1,node2): ", sess.run([node1, node2]))
print("sess.run(node3): ", sess.run(node3))
