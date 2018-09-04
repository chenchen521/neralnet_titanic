import os
import numpy as np
import pandas as pd
import tensorflow as tf

train_data = pd.read_csv('D:/Downloads/kaggle_titanic_dataset/train.csv')

from sklearn.ensemble import RandomForestRegressor
age = train_data[['Age','Survived','Fare','Parch','SibSp','Pclass']]
age_notnull = age.loc[(train_data.Age.notnull())]
age_isnull = age.loc[(train_data.Age.isnull())]
X = age_notnull.values[:,1:]
Y = age_notnull.values[:,0]
rfr = RandomForestRegressor(n_estimators=1000,n_jobs=-1)
rfr.fit(X,Y)
predictAges = rfr.predict(age_isnull.values[:,1:])
train_data.loc[(train_data.Age.isnull()),'Age'] = predictAges

train_data = train_data.fillna(0) #缺失字段填0

train_data.loc[train_data['Sex']=='male','Sex'] = 0
train_data.loc[train_data['Sex']=='female','Sex'] = 1

train_data['Embarked'] = train_data['Embarked'].fillna('S')
train_data.loc[train_data['Embarked'] == 'S','Embarked'] = 0
train_data.loc[train_data['Embarked'] == 'C','Embarked'] = 1
train_data.loc[train_data['Embarked'] == 'Q','Embarked'] = 2

train_data.drop(['Cabin'],axis=1,inplace=True)
train_data['Deceased'] = train_data['Survived'].apply(lambda s: 1 - s)

dataset_X = train_data[['Sex','Age','Pclass','SibSp','Parch','Fare']]
dataset_Y = train_data[['Deceased','Survived']]

#分为测试集和验证集
from sklearn.model_selection import train_test_split
#(712,6) (712,2)
X_train, X_val, Y_train, Y_val = train_test_split(dataset_X.as_matrix(), dataset_Y.as_matrix(), test_size = 0.2,random_state = 42)

x = tf.placeholder(tf.float32,shape = [None,6],name = 'input')
y = tf.placeholder(tf.float32,shape = [None,2],name = 'label')
weights1 = tf.Variable(tf.random_normal([6,6]),name = 'weights1')
bias1 = tf.Variable(tf.zeros([6]),name = 'bias1')
a = tf.nn.relu(tf.matmul(x,weights1) + bias1)                                   #(712,6)

weights2 = tf.Variable(tf.random_normal([6,2]),name = 'weights2')               #(6,2)
bias2 = tf.Variable(tf.zeros([2]),name = 'bias2')                               #(1,2)
z = tf.matmul(a,weights2) + bias2                                               #(712,2)                                          
y_pred = tf.nn.softmax(z)                                                       #(712,2)                    

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=z)) #
correct_pred = tf.equal(tf.argmax(y,1),tf.argmax(y_pred,1)) #boolean序列
acc_op = tf.reduce_mean(tf.cast(correct_pred,tf.float32))   #准确率
train_op = tf.train.AdamOptimizer(0.001).minimize(cost)     #训练函数

saver = tf.train.Saver()                              

ckpt_dir = './ckpt_dir'

if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.latest_checkpoint(ckpt_dir)
    if ckpt:
        print('Restoring from checkpoint: %s' %ckpt)
        saver.restore(sess, ckpt)
    for epoch in range(30):
        total_loss = 0.
        for i in range(len(X_train)):
            feed_dict = {x:[X_train[i]], y:[Y_train[i]]}
            _, loss = sess.run([train_op, cost], feed_dict = feed_dict)
            total_loss += loss
        print('Epoch:%4d, total loss = %.12f' %(epoch, total_loss))
        if epoch % 10 == 0:
            accuracy = sess.run(acc_op, feed_dict={x:X_val, y:Y_val})
            print('Accuracy on validation set: %.9f' % accuracy)
            saver.save(sess, ckpt_dir + '/logistic.ckpt')
    print('training complete!')
    accuracy = sess.run(acc_op, feed_dict = {x:X_val, y: Y_val})
    print('Accuracy on validation set: %.9f' % accuracy)
    
    pred = sess.run(y_pred, feed_dict = {x:X_val})
    correct = np.equal(np.argmax(pred, 1), np.argmax(Y_val,1))
    numpy_accuracy = np.mean(correct.astype(np.float32))
    print("Accuracy on validation set(numpy): %.9f" % numpy_accuracy)
    saver.save(sess, ckpt_dir + '/logistic.ckpt')
    
    
    
    
    
    


