# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import csv

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" #for training on gpu

# Create dictionary of target classes
label_dict = {
 0: 'neutral',
 1: 'anger',
 2: 'contempt',
 3: 'disgust',
 4: 'fear',
 5: 'happy',
 6: 'sadness',
 7: 'surprise'
}

# total classes (digits)
n_classes = 8

def load_data(data_directory):
    directories = [d for d in os.listdir(data_directory) 
                   if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    images = []
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f) 
                      for f in os.listdir(label_directory) 
                      if f.endswith(".jpg")]
        for f in file_names:
            image = cv2.imread(f,0)
            images.append(image)
            
            label_num = [key for key, value in label_dict.items() if value == d]
            labels.append(label_num)
    
    return np.array(images), np.array(labels)

ROOT_PATH = "dataset"
train_data_directory = os.path.join(ROOT_PATH, "training")
test_data_directory = os.path.join(ROOT_PATH, "testing")

train_X, train_Y = load_data(train_data_directory)
test_X, test_Y = load_data(test_data_directory)

# One-hot coding labels
train_Y = train_Y.reshape(-1)
train_Y = np.eye(n_classes)[train_Y]

test_Y = test_Y.reshape(-1)
test_Y = np.eye(n_classes)[test_Y]

""" # Print the `images` dimensions
print(train_X.ndim)

# Print the number of `images`'s elements
print(train_X.shape)

 # Print the `images` dimensions
print(train_Y.ndim)

# Print the number of `images`'s elements
print(train_Y.size)

# Print the `images` dimensions
print(test_X.ndim)

# Print the number of `images`'s elements
print(test_X.shape)

 # Print the `images` dimensions
print(test_Y.ndim)

# Print the number of `images`'s elements
print(test_Y.size)

 """
""" # Shapes of training set
print("Training set (images) shape: {shape}".format(shape=train_X.shape))
print("Training set (labels) shape: {shape}".format(shape=train_Y.shape))

# Shapes of test set
print("Test set (images) shape: {shape}".format(shape=test_X.shape))
print("Test set (labels) shape: {shape}".format(shape=test_Y.shape))
 """
""" 
plt.figure(figsize=[5,5])

# Display the first image in training data
plt.subplot(121)
curr_img = np.reshape(train_X[0], (28,28))
curr_lbl = np.argmax(train_Y[0,:])
plt.imshow(curr_img, cmap='gray')
plt.title("(Label: " + str(label_dict[curr_lbl]) + ")")

# Display the first image in testing data
plt.subplot(122)
curr_img = np.reshape(test_X[0], (28,28))
curr_lbl = np.argmax(test_Y[0,:])
plt.imshow(curr_img, cmap='gray')
plt.title("(Label: " + str(label_dict[curr_lbl]) + ")")

plt.show() """

training_iters = 10001
learning_rate = 0.0001
batch_size = 16

# data input (img shape: 227*227)
n_input = 227

# Reshape training and testing image
train_X = train_X.reshape(-1, n_input, n_input, 1)
test_X = test_X.reshape(-1,n_input,n_input,1)

# Rescale training and testing data
train_X = train_X.astype('float32')
test_X = test_X.astype('float32')

train_X = train_X/255
test_X = test_X/255

train_Y = train_Y.reshape(-1, n_classes)
test_Y = test_Y.reshape(-1, n_classes)

#both placeholders are of type float
x = tf.placeholder("float", [None, 227,227,1])
y = tf.placeholder("float", [None, n_classes])

def conv2d(x, W, b, strides=1, padding='SAME'):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding=padding)
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x) 

def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME')

weights = {
    'wc1': tf.get_variable('W0', shape=(11,11,1,96), initializer=tf.contrib.layers.xavier_initializer()), 
    'wc2': tf.get_variable('W1', shape=(3,3,96,128), initializer=tf.contrib.layers.xavier_initializer()), 
    'wc3': tf.get_variable('W2', shape=(3,3,128,256), initializer=tf.contrib.layers.xavier_initializer()), 
    'wc4': tf.get_variable('W3', shape=(2,2,256,256), initializer=tf.contrib.layers.xavier_initializer()), 
    'wd1': tf.get_variable('W4', shape=(4*4*256,512), initializer=tf.contrib.layers.xavier_initializer()), 
    'out': tf.get_variable('W5', shape=(512,n_classes), initializer=tf.contrib.layers.xavier_initializer()), 
}
biases = {
    'bc1': tf.get_variable('B0', shape=(96), initializer=tf.contrib.layers.xavier_initializer()),
    'bc2': tf.get_variable('B1', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
    'bc3': tf.get_variable('B2', shape=(256), initializer=tf.contrib.layers.xavier_initializer()),
    'bc4': tf.get_variable('B3', shape=(256), initializer=tf.contrib.layers.xavier_initializer()),
    'bd1': tf.get_variable('B4', shape=(512), initializer=tf.contrib.layers.xavier_initializer()),
    'out': tf.get_variable('B5', shape=(n_classes), initializer=tf.contrib.layers.xavier_initializer()),
}
strides = {
    'sc1': 4,
    'sc2': 2,
    'sc3': 1,
    'sc4': 2
}
paddings={
    'pc1': 'VALID',
    'pc2': 'VALID',
    'pc3': 'VALID',
    'pc4': 'VALID'
}

def conv_net(x, weights, biases):  

    # here we call the conv2d function we had defined above and pass the input image x, weights wc1 and bias bc1.
    conv1 = conv2d(x, weights['wc1'], biases['bc1'], strides['sc1'], paddings['pc1'])
    print(f'conv1.shape {conv1.shape}')
    # conv1 = tf.nn.dropout(conv1, 0.9)

    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'], strides['sc2'])
    print(f'conv2.shape {conv2.shape}')
    # conv2 = tf.nn.dropout(conv2, 0.9)

    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window
    pool1 = maxpool2d(conv2, k=2)
    print(f'pool1.shape {pool1.shape}')
    # pool1 = tf.nn.dropout(pool1, 0.9)

    conv3 = conv2d(pool1, weights['wc3'], biases['bc3'], strides['sc3'])
    print(f'conv3.shape {conv3.shape}')
    # conv3 = tf.nn.dropout(conv3, 0.9)

    conv4 = conv2d(conv3, weights['wc4'], biases['bc4'], strides['sc4'])
    print(f'conv4.shape {conv4.shape}')
    # conv4 = tf.nn.dropout(conv4, 0.9)

    pool2 = maxpool2d(conv4, k=2)
    print(f'pool2.shape {pool2.shape}')
    # pool2 = tf.nn.dropout(pool2, 0.9)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(pool2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, 0.5)

    # Output, class prediction
    # finally we multiply the fully connected layer with the weights and add a bias term. 
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

pred = conv_net(x, weights, biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#Here you check whether the index of the maximum value of the predicted image is equal to the actual labelled image. and both will be a column vector.
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

#calculate accuracy across all the given images and average them out. 
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init) 
    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []
    summary_writer = tf.summary.FileWriter('./Output', sess.graph)
    for i in range(training_iters):
        for batch in range(len(train_X)//batch_size):
            batch_x = train_X[batch*batch_size:min((batch+1)*batch_size,len(train_X))]
            batch_y = train_Y[batch*batch_size:min((batch+1)*batch_size,len(train_Y))]    
            # Run optimization op (backprop).
                # Calculate batch loss and accuracy
            opt = sess.run(optimizer, feed_dict={x: batch_x,
                                                              y: batch_y})
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y})
        print("Iter " + str(i) + ", Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
        print("Optimization Finished!")

        # Calculate accuracy for all test images
        test_acc,valid_loss = sess.run([accuracy,cost], feed_dict={x: test_X,y : test_Y})
        train_loss.append(loss)
        test_loss.append(valid_loss)
        train_accuracy.append(acc)
        test_accuracy.append(test_acc)
        print("Testing Accuracy:","{:.5f}".format(test_acc)) 
    summary_writer.close()

plt.plot(range(len(train_loss)), train_loss, 'b', label='Training loss')
plt.plot(range(len(train_loss)), test_loss, 'r', label='Test loss')
plt.title('Training and Test loss')
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.legend()
plt.figure()
plt.show()

plt.plot(range(len(train_loss)), train_accuracy, 'b', label='Training Accuracy')
plt.plot(range(len(train_loss)), test_accuracy, 'r', label='Test Accuracy')
plt.title('Training and Test Accuracy')
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.legend()
plt.figure()
plt.show() 

# output csv
rows = []
rows.append(['Epoch','Training loss','Testing loss','Training Accuracy','Test Accuracy'])
for i in range(len(train_loss)):
    rows.append([i+1, train_loss[i], test_loss[i], train_accuracy[i], test_accuracy[i]])

with open("output.csv",'w',newline='') as resultFile:
    wr = csv.writer(resultFile, dialect='excel')
    wr.writerows(rows)