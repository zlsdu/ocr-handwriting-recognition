# coding=utf-8
import tensorflow as tf
import os
import glob
import random
import numpy as np
from PIL import Image
from PIL import ImageFilter



def transform(im, flag=True):
    '''
    将传入的图片进行预处理：对图像进行图像缩放和数据增强
    Args:
        im :　传入的待处理的图片
    Return：
        graph : 返回经过预处理的图片
    #random.uniform(a, b)随机产生[a, b)之间的一个浮点数
    '''
    graph=np.zeros(graphSize[1]*graphSize[0]*1).reshape(graphSize[0],graphSize[1],1)
    deltaX=0
    deltaY=0
    ratio=1.464
    if flag:
        lowerRatio=max(1.269,im.size[1]*1.0/graphSize[0],im.size[0]*1.0/graphSize[1])
        upperRatio=max(lowerRatio,2.0)
        ratio=random.uniform(lowerRatio,upperRatio)
        deltaX=random.randint(0,int(graphSize[0]-im.size[1]/ratio))
        deltaY=random.randint(0,int(graphSize[1]-im.size[0]/ratio))
    else:
        ratio=max(1.464,im.size[1]*1.0/graphSize[0],im.size[0]*1.0/graphSize[1])
        deltaX=int(graphSize[0]-im.size[1]/ratio)>>1
        deltaY=int(graphSize[1]-im.size[0]/ratio)>>1
    height=int(im.size[1]/ratio)
    width=int(im.size[0]/ratio)
    data = im.resize((width,height),Image.ANTIALIAS).getdata()
    data = 1-np.asarray(data,dtype='float')/255.0
    data = data.reshape(height,width)
    graph[deltaX:deltaX+height,deltaY:deltaY+width,0]=data
    return graph


def countMargin(v,minSum,direction=True):
    '''
    将向量v中的数和minSum作比较，#目前猜测返回向量v中比minSum小的数的数量或者返回的是索引
    Args:
        v = list
        minSum = Int
    Return:
        v中比minSum小的项数
    '''
    if direction:
        for i in range(len(v)):
            if v[i]>minSum:
                return i
        return len(v)
    for i in range(len(v)-1,-1,-1):
        if v[i]>minSum:
            return len(v)-i-1
    return len(v)


def splitLine(seg,dataSum,h,maxHeight):
    i=0
    while i<len(seg)-1:
        if seg[i+1]-seg[i]<maxHeight:
            i+=1
            continue
        x=countMargin(dataSum[seg[i]:],3,True)
        y=countMargin(dataSum[:seg[i+1]],3,False)
        if seg[i+1]-seg[i]-x-y<maxHeight:
            i+=1
            continue
        idx=dataSum[seg[i]+x+h:seg[i+1]-h-y].argmin()+h
        if 0.33<=idx/(seg[i+1]-seg[i]-x-y)<=0.67:
            seg.insert(i+1,dataSum[seg[i]+x+h:seg[i+1]-y-h].argmin()+seg[i]+x+h)
        else:
            i+=1


def getLine(im,data,upperbound=8,lowerbound=25,threshold=30,h=40,minHeight=35,maxHeight=120,beginX=20,endX=-20,beginY=140,endY=1100,merged=True):
    dataSum=data[:,beginX:endX].sum(1)
    lastPosition=beginY
    seg=[]
    flag=False
    cnt=0
    for i in range(beginY,endY):
        if dataSum[i]<=lowerbound:
            flag=True
            if dataSum[i]<=upperbound:
                cnt=0
                continue
        if flag:
            cnt+=1
            if cnt>=threshold:
                lineNo=np.argmin(dataSum[lastPosition:i])+lastPosition if threshold<=i-beginY else beginY
                if not merged or len(seg)==0 or lineNo-seg[-1]-countMargin(dataSum[seg[-1]:],5,True)-countMargin(dataSum[:lineNo],5,False)>minHeight:
                    seg.append(lineNo)
                else:
                    avg1=dataSum[max(0,seg[-1]-1):seg[-1]+2]
                    avg1=avg1.sum()/avg1.shape[0]
                    avg2=dataSum[max(0,lineNo-1):lineNo+2]
                    avg2=avg2.sum()/avg2.shape[0]
                    if avg1>avg2:
                        seg[-1]=lineNo
                lastPosition=i
                flag=False
    lineNo=np.argmin(dataSum[lastPosition:]>10)+lastPosition if threshold<i else beginY
    if not merged or len(seg)==0 or lineNo-seg[-1]-countMargin(dataSum[seg[-1]:],10,True)-countMargin(dataSum[:lineNo],10,False)>minHeight:
        seg.append(lineNo)
    else:
        avg1=dataSum[max(0,seg[-1]-1):seg[-1]+2]
        avg1=avg1.sum()/avg1.shape[0]
        avg2=dataSum[max(0,lineNo-1):lineNo+2]
        avg2=avg2.sum()/avg2.shape[0]
        if avg1>avg2:
            seg[-1]=lineNo
    splitLine(seg,dataSum,h,maxHeight)
    results=[]
    for i in range(0,len(seg)-1):
        results.append(im.crop((0,seg[i]+countMargin(dataSum[seg[i]:],0),im.size[0],seg[i+1]-countMargin(dataSum[:seg[i+1]],0,False))))
    return results


def calEditDistance(text1, text2):
    dp=np.asarray([0]*(len(text1)+1)*(len(text2)+1)).reshape(len(text1)+1,len(text2)+1)
    dp[0]=np.arange(len(text2)+1)
    dp[:,0]=np.arange(len(text1)+1)
    for i in range(1,len(text1)+1):
        for j in range(1,len(text2)+1):
            if text1[i-1]==text2[j-1]:
                dp[i,j]=dp[i-1,j-1]
            else:
                dp[i,j]=min(dp[i,j-1],dp[i-1,j],dp[i-1,j-1])+1
    return dp[-1,-1]



#chr函数： 将数字转化成字符
#ord函数： 将字符转化成数字
#characterNo字典：a-z, A-Z, 0-10, " .,?\'-:;!/\"<>&(+" 为key分别对应值是0-25,26-51,52-61,62...
#characters列表： 存储的是cahracterNo字典的key
#建立characterNo字典的意思是： 为了将之后手写体对应的txt文件中的句子转化成 数字编码便于存储和运算求距离
charactersNo={}
characters=[]
length=[]
graphSize=(112,1024)

for i in range(26):
    charactersNo[chr(ord('a')+i)]=i
    characters.append(chr(ord('a')+i))
for i in range(26):
    charactersNo[chr(ord('A')+i)]=i+26
    characters.append(chr(ord('A')+i))
for i in range(10):
    charactersNo[chr(ord('0')+i)]=i+52
    characters.append(chr(ord('0')+i))
punctuations=" .,?\'-:;!/\"<>&(+"
for p in punctuations:
    charactersNo[p]=len(charactersNo)
    characters.append(p)


#读取了train_img和train_txt文件夹下的所有文件的读取路径
#下面代码的作用是： 
#Imgs:列表结构 存储的是手写的英文图片
#Y: 数组结构 存储的是图片对应的txt文件中句子，只不过存储的是字符转码后的数字
#length: 数组结构 存储的是图片对应的txt文件中句子含有字符的数量
imgFiles=glob.glob(os.path.join("train_img", "*"))
imgFiles.sort()
txtFiles=glob.glob(os.path.join("train_txt", "*"))
txtFiles.sort()
Imgs=[]
Y=[]
length=[]
for i in range(len(imgFiles)):
    fin=open(txtFiles[i])
    line=fin.readlines()
    line=line[0]
    fin.close()
    y=np.asarray([0]*(len(line)))
    succ=True
    for j in range(len(line)):
        if line[j] not in charactersNo:
            succ=False
            break
        y[j]=charactersNo[line[j]]
    if not succ:
        continue
    Y.append(y)
    length.append(len(line))
    im = Image.open(imgFiles[i])
    width,height = im.size#1499,1386
    im = im.convert("L")
    Imgs.append(im)


#np.asarray()函数 和 np.array()函数： 将list等结构转化成数组
#区别是np.asarray()函数不是copy对象，而np.array()函数是copy对象
print("train:",len(Imgs),len(Y))
Y = np.asarray(Y)
length = np.asarray(length)


def create_sparse(Y,dtype=np.int32):
    '''
    对txt文本转化出来的数字序列Y作进一步的处理
    Args:
        Y
    Return:
        indices: 数组Y下标索引构成的新数组
        values: 下标索引对应的真实的数字码
        shape
    '''
    indices = []
    values = []
    for i in range(len(Y)):
        for j in range(len(Y[i])):
            indices.append((i,j))
            values.append(Y[i][j])

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(Y), np.asarray(indices).max(0)[1] + 1], dtype=np.int64) #[64,180]

    return (indices, values, shape)


#以下经过四层卷积神经网络处理  四层卷积池化
conv1_filter=32
conv2_filter=64
conv3_filter=128
conv4_filter=256

inputs = tf.placeholder(tf.float32, shape=[None,graphSize[0],graphSize[1],1])
W_conv1 = tf.Variable(tf.truncated_normal(([3, 3, 1, conv1_filter]),stddev=0.1,dtype=tf.float32), name="W_conv1")
b_conv1 = tf.Variable(tf.constant(0., shape=[conv1_filter],dtype=tf.float32), name="b_conv1")
h_conv1 = tf.nn.relu(tf.nn.conv2d(inputs, W_conv1, strides=[1,1,1,1], padding='SAME') + b_conv1)
h_pool1 = tf.nn.max_pool(h_conv1,ksize=[1,2,2,1],strides=[1,2,2,1], padding='VALID')
W_conv2 = tf.Variable(tf.truncated_normal(([5, 5, conv1_filter, conv2_filter]),stddev=0.1), name="W_conv2")
b_conv2 = tf.Variable(tf.constant(0., shape=[conv2_filter]), name="b_conv2")
keep_prob = tf.placeholder(tf.float32)     #防止过拟合
h_conv2 = tf.nn.relu(tf.nn.conv2d(tf.nn.dropout(h_pool1,keep_prob), W_conv2, strides=[1,1,1,1], padding='SAME') + b_conv2)
h_pool2 = tf.nn.max_pool(h_conv2,ksize=[1,2,1,1],strides=[1,2,1,1], padding='VALID')
W_conv3 = tf.Variable(tf.truncated_normal(([5, 5, conv2_filter, conv3_filter]),stddev=0.1), name="W_conv3")
b_conv3 = tf.Variable(tf.constant(0., shape=[conv3_filter]), name="b_conv3")
h_conv3 = tf.nn.relu(tf.nn.conv2d(tf.nn.dropout(h_pool2,keep_prob), W_conv3, strides=[1,1,1,1], padding='SAME') + b_conv3)
h_pool3 = tf.nn.max_pool(h_conv3,ksize=[1,4,2,1],strides=[1,4,2,1], padding='VALID')
W_conv4 = tf.Variable(tf.truncated_normal(([5, 5, conv3_filter, conv4_filter]),stddev=0.1), name="W_conv4")
b_conv4 = tf.Variable(tf.constant(0., shape=[conv4_filter]), name="b_conv4")
h_conv4 = tf.nn.relu(tf.nn.conv2d(tf.nn.dropout(h_pool3,keep_prob), W_conv4, strides=[1,1,1,1], padding='SAME') + b_conv4)
h_pool4 = tf.nn.max_pool(h_conv4,ksize=[1,7,1,1],strides=[1,7,1,1], padding='VALID')


rnn_inputs=tf.reshape(tf.nn.dropout(h_pool4, keep_prob),[-1,256,conv4_filter])

num_hidden=512
num_classes=len(charactersNo)+1
W = tf.Variable(tf.truncated_normal([num_hidden,num_classes],stddev=0.1), name="W")
b = tf.Variable(tf.constant(0., shape=[num_classes]), name="b")
global_step = tf.Variable(0, trainable=False)#全局步骤计数

seq_len = tf.placeholder(tf.int32, shape=[None])
labels=tf.sparse_placeholder(tf.int32, shape=[None,2])
cell_fw = tf.nn.rnn_cell.LSTMCell(num_hidden>>1, state_is_tuple=True)
cell_bw = tf.nn.rnn_cell.LSTMCell(num_hidden>>1, state_is_tuple=True)
outputs_fw_bw, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, rnn_inputs, seq_len, dtype=tf.float32)
outputs1 = tf.concat(outputs_fw_bw, 2)

shape = tf.shape(inputs)
batch_s, max_timesteps = shape[0], shape[1]
outputs = tf.reshape(outputs1, [-1, num_hidden])

logits0 = tf.matmul(tf.nn.dropout(outputs,keep_prob), W) + b
logits1 = tf.reshape(logits0, [batch_s, -1, num_classes])
logits = tf.transpose(logits1, (1, 0, 2))
logits = tf.cast(logits, tf.float32)

loss = tf.nn.ctc_loss(labels, logits, seq_len)
cost = tf.reduce_mean(loss)
optimizer1 = tf.train.AdamOptimizer(learning_rate=0.0003).minimize(cost, global_step=global_step)
optimizer2 = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost, global_step=global_step)
width1_decoded, width1_log_prob=tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False,beam_width=1)
decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)
width1_acc = tf.reduce_mean(tf.edit_distance(tf.cast(width1_decoded[0], tf.int32), labels))
acc = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), labels))
nBatchArray=np.arange(Y.shape[0])
epoch=100
batchSize=32
saver=tf.train.Saver(max_to_keep=1)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.Session(config=config)
bestDevErr=100.0
with sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, "model/model.ckpt")
    print(outputs.get_shape())
    for ep in range(epoch):
        np.random.shuffle(nBatchArray)
        for i in range(0, Y.shape[0], batchSize):
            batch_output = create_sparse(Y[nBatchArray[i:i+batchSize]])
            X=[None]*min(Y.shape[0]-i,batchSize)
            for j in range(len(X)):
                #将图像转化成[112, 1024], 在输入网络前图像通过im.getdata()获取图像信息，已经强制转化成[112, 1024]的ndarray对象
                X[j]=transform(Imgs[nBatchArray[i+j]])

            feed_dict={inputs:X,seq_len :np.ones(min(Y.shape[0]-i,batchSize)) * 256,labels:batch_output,keep_prob:0.6}
            if ep<50:
                sess.run(optimizer1, feed_dict=feed_dict)
            else:
                sess.run(optimizer2, feed_dict=feed_dict)
            print(ep,i,"loss:",tf.reduce_mean(loss.eval(feed_dict=feed_dict)).eval(),"err:",tf.reduce_mean(width1_acc.eval(feed_dict=feed_dict)).eval())
        #saver.save(sess,"model/model.ckpt")