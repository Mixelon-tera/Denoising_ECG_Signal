import tensorflow as tf
import numpy as np
import os
import utils as ut

n_visible = 3600
n_hidden = 1800
corruption_level =0.3

#create node for input layer
X = tf.placeholder("float",[None, n_visible], name='X')

#create node for corrupt layer
mask = tf.placeholder("float", [None, n_visible], name='mask')

#create nodes for hidden layer (number hidden node = 1800)
W_init_max = 4 * np.sqrt(6. / (n_visible + n_hidden))
W_init = tf.random_uniform(shape=[n_visible,n_hidden],
                           minval=-W_init_max,
                           maxval=W_init_max)

W = tf.Variable(W_init, name='W')
b = tf.Variable(tf.zeros([n_hidden]), name='b')

W_prime = tf.transpose(W) # tied weight
b_prime = tf.Variable(tf.zeros([n_visible]), name='b_prime')

def model(X, mask, W, b, W_prime, b_prime):
    tilde_X = mask * X
    print("tildeshape is %s" %str(tilde_X.shape))
    Y= tf.nn.sigmoid(tf.matmul(tilde_X,W)+b) #hidden state
    Z = tf.nn.sigmoid(tf.matmul(Y, W_prime)+b_prime) #reconstructed
    return Z

# build model graph
Z = model(X, mask, W, b, W_prime, b_prime)

#create cost function
cost = tf.reduce_sum(tf.pow(X - Z, 2)) #minimize squared error
train_op = tf.train.GradientDescentOptimizer(0.02).minimize(cost)

filename=[]
noisename=[]
#load data from util
for file in os.listdir("/Users/Mix_Tera/Desktop/ECG_Tera/database/mitdb/"):
    if file.endswith(".csv"):
        filename.append(file)
for file in os.listdir("/Users/Mix_Tera/Desktop/ECG_Tera/database/nstdb/"):
    if file.endswith(".csv"):
        noisename.append(file)
with tf.Session() as sess:
        #Initialize all variables
        #tf.global_variables_initializer()
        tf.initialize_all_variables().run()
        for fn in filename:
            for nn in noisename:
                time , train_data = ut.combine(fn,1,nn,2)
                #ut.plotdata(time,train_data)
                input_ = train_data[:]
                input_ = np.asarray(input_)
                mask_np = np.random.binomial(1,1-corruption_level,input_.shape)
                sess.run(train_op, feed_dict={X: input_, mask: mask_np})
                print("success!!!")
                #print(mask_np.shape)



