import numpy as np
import os
import tensorflow as tf
import utils as ut
import random
import matplotlib.pyplot as plt
import detect_peaks as dp
# NETOWRK PARAMETERS
n_input    = 270
n_hidden_1 = 135
n_hidden_2 = 70
n_output   = 270
n_sample =  2160#1360
n_test = 800
epochs     = 1500
batch_size = 10
disp_step  = 10

text_result = open("result_ecg.txt", "w")

print ("PACKAGES LOADED")
global_time =[]
i=0
j=0
train_egs = np.zeros((1, n_input), dtype=np.float64)
test_egs = np.zeros((1, n_input), dtype=np.float64)
#get training data
for file in os.listdir("/Users/Mix_Tera/Desktop/ECG_Tera/database/mitdb/train/"):
    if file.endswith(".csv"):
        time , train_data  = ut.getdataset(file,0)
        global_time=time
        if i==0:
            train_egs = train_data
        else:
            train_egs = np.concatenate((train_egs, train_data), axis=0)
        i+=1
#get test data
for file in os.listdir("/Users/Mix_Tera/Desktop/ECG_Tera/database/mitdb/test/"):
    if file.endswith(".csv"):
        time , test_data   = ut.getdataset(file,1)
        if j==0:
            test_egs = test_data
        else:
            test_egs = np.concatenate((test_egs, test_data), axis=0)
        j+=1

train_egs = random.sample(train_egs, len(train_egs))
print ("ECG LOADED")

# PLACEHOLDERS
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_output])
dropout_keep_prob = tf.placeholder("float")

# WEIGHTS
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1]), name="h1"),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]), name="h2"),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_output]), name="wout")
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1]), name="b1"),
    'b2': tf.Variable(tf.random_normal([n_hidden_2]), name="b2"),
    'out': tf.Variable(tf.random_normal([n_output]), name="bout")
}

encode_in = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['h1']),biases['b1']))

encode_out = tf.nn.dropout(encode_in, dropout_keep_prob)

decode_in = tf.nn.sigmoid(tf.add(tf.matmul(encode_out, weights['h2']),biases['b2']))

decode_out = tf.nn.dropout(decode_in,dropout_keep_prob)

y_pred = tf.nn.sigmoid(tf.matmul(decode_out,weights['out']) +biases['out'])

# COST
cost = tf.sqrt(tf.reduce_mean(tf.pow(y_pred - y, 2)))
# SNR
snr = 10.0 * tf.log( tf.reduce_sum(tf.pow(y,2)) / tf.reduce_sum(tf.pow(y_pred-y,2)) )
# OPTIMIZER
optmizer = tf.train.AdamOptimizer(0.002).minimize(cost)

# INITIALIZER
init = tf.global_variables_initializer()
saver = tf.train.Saver()
# Launch the graph
with tf.Session() as sess:
    #sess.run(init)

    saver.restore(sess, "/Users/Mix_Tera/Desktop/ECG_Tera/models/model16.ckpt")
    print("Model restored.")

    print ("Start Training")
    print("get noise")
    noisy = ut.getallnoise()
    for epoch in range(epochs):

        train_egs = random.sample(train_egs, len(train_egs))

        num_batch  = int(n_sample/batch_size)
        total_cost = 0.
        total_snr = 0.
        print("start loop")
        for i in range(num_batch):
            '''
            batch_xs = np.reshape(train_egs[i], (1, 270))
            #ut.plotecg(np.reshape(batch_xs, (1440,)).tolist(),global_time)
            k = int(np.floor(np.random.uniform(0,160)))
            batch_xs_noisy = batch_xs + 0.3*noisy[k] + 0.2*(noisy[k + 160]+ noisy[k + 320])
            #ut.plotecg(np.reshape(batch_xs_noisy, (1440,)).tolist(), global_time)
            coolbatch = np.reshape(ut.cooldown(batch_xs,0),(1,270))
            coolnoisy = np.reshape(ut.cooldown(batch_xs_noisy,1),(1,270))
            feeds = {x: coolnoisy, y: coolbatch, dropout_keep_prob: 0.8}
            sess.run(optmizer, feed_dict=feeds)
            total_cost += sess.run(cost, feed_dict=feeds)
            '''
        print("finish loop")
        # DISPLAY
        print("epoch = %d" %epoch)
        #global_time = ut.gettime()
        if epoch !=0 and epoch%1==0:
            #print ("Epoch %02d/%02d average cost: %.10f"
                   #% (epoch, epochs, total_cost/num_batch))

            print ("Start Test")
            #randidx   = np.random.randint(test_egs.shape[0], size=1)
            randidx = [3]
            orgvec    = test_egs[randidx, :]
            testvec   = test_egs[randidx, :]
            k = int(np.floor(np.random.uniform(0, 160)))
            #noisyvec = testvec +  0.3*noisy[k] + 0.2*(noisy[k + 160]+ noisy[k + 320])
            noisyvec = testvec + 0.3 * noisy[k+160]
            coolnoisyvec = np.reshape(ut.cooldown(noisyvec,1),(1,270))
            outvec   = sess.run(y_pred,feed_dict={x:coolnoisyvec,dropout_keep_prob: 1})

            feeds = {x: outvec, y: orgvec, dropout_keep_prob: 1}
            total_cost += sess.run(cost, feed_dict=feeds)
            total_snr += sess.run(snr, feed_dict=feeds)
            print ("Epoch %02d/%02d average cost: %.10f"
                   % (epoch, epochs, total_cost / num_batch))
            
            text_result.write("Epoch %02d/%02d average cost: %.10f"
                              % (epoch, epochs, total_cost / num_batch))

            print ("Signal to noise ratio: %.10f"
                   % total_snr)

            text_result.write("Signal to noise ratio: %.10f"
                   % total_snr)

            if epoch%1==0:

                f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
                ax1.plot(global_time, np.reshape(orgvec, (270,)).tolist(), color='g')
                ax1.set_title('ECG signal')
                ax2.plot(global_time, ut.normalization(np.reshape(noisyvec, (270,)).tolist()), color='r')
                ax3.plot(global_time, ut.normalization(np.reshape(outvec, (270,))), color='b')
                plt.show()
                '''
                ut.plotecg(np.reshape(orgvec, (270,)).tolist(), global_time)
                ut.plotecg(ut.normalization(np.reshape(noisyvec, (270,)).tolist()), global_time)
                ut.plotecg(ut.normalization(np.reshape(outvec, (270,)).tolist()), global_time)
                '''
                #outvec = ut.cooldown(outvec,0))

                #ut.multiplotecg(np.reshape(orgvec, (270,)).tolist(),
                                #ut.normalization(np.reshape(noisyvec, (270,)).tolist()),
                               #ut.normalization(np.reshape(outvec, (270,))),global_time)

    #save_path = saver.save(sess, "/Users/Mix_Tera/Desktop/ECG_Tera/models/model24.ckpt")
    #print("Model saved in file: %s" % save_path)
text_result.close()


