import tensorflow as tf

dataset = tf.data.Dataset.range(10)
# dataset = dataset.repeat(2)
# dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(10))
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()
result = tf.where(tf.greater_equal(next_element,5),next_element,0)
with tf.Session() as sess:
    for i in range(10):
      value = sess.run(result)
      print("{reslt}",value)
    # value = sess.run(result)
    # print("{reslt}",value)