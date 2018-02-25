#%%
from keras.applications import VGG16
from keras import backend as K
import tensorflow as tf

#K.set_learning_phase(0)
mn = VGG16()
saver = tf.train.Saver()
sess = K.get_session()
sess.run(tf.global_variables_initializer())
saver.save(sess, "./TF_Model/vgg16")

fw = tf.summary.FileWriter('logs', sess.graph)
fw.close()

# mvNCProfile TF_Model/vgg16.meta -in=input_1 -on=predictions/Softmax -s 12