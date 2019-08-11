    # =========================================================================
import tensorflow as tf
import config as CF
    
class model(object):
    def __init__(self,sess,config,images_batch,labels):
        self.sess=sess
        self.train_writer = tf.summary.FileWriter(config["logs_train_dir"], sess.graph)
        self.model_type=CF.config["num"]
        
        self.batch_size=config["BATCH_SIZE"]
        self.n_classes=config["N_CLASSES"]
        self.learning_rate=config["learning_rate"]
        #self.saver = tf.train.Saver()
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)
        self.build(images_batch,labels)
# 训练操作定义  
    def build(self,images_batch,labels):
        if self.model_type==1:
            print("选用模型1")
            self.structure_1(images_batch)
        elif self.model_type==2:
            print("选用模型2")
            self.structure_2(images_batch)
        elif self.model_type==3:
            print("选用模型3")
            self.is_training = tf.placeholder(tf.bool)
            self.is_use_l2 = tf.placeholder(tf.bool)
            self.lam = tf.placeholder(tf.float32)
            self.keep_prob = tf.placeholder(tf.float32)
            self.structure_3(images_batch)
        self.evaluation(labels)
        self.losses(labels)
        self.opt()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
    # =========================================================================
    # 网络结构定义
    # 输入参数：images，image batch、4D tensor、tf.float32、[batch_size, width, height, channels]
    # 返回参数：logits, float、 [batch_size, n_classes]
    
    def structure_1(self,images):
        with tf.variable_scope('conv1') as scope:
            weights = tf.Variable(tf.truncated_normal(shape=[3, 3, 3, 64], stddev=1.0, dtype=tf.float32),
                                  name='weights', dtype=tf.float32)
    
            biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[64]),
                                 name='biases', dtype=tf.float32)
    
            conv = tf.nn.conv2d(images, weights, strides=[1, 1, 1, 1], padding='SAME')
            pre_activation = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(pre_activation, name=scope.name)
        with tf.variable_scope('pooling1_lrn') as scope:
            pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pooling1')
            norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
    
        with tf.variable_scope('conv2') as scope:
            weights = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 16], stddev=0.1, dtype=tf.float32),
                                  name='weights', dtype=tf.float32)
    
            biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[16]),
                                 name='biases', dtype=tf.float32)
    
            conv = tf.nn.conv2d(norm1, weights, strides=[1, 1, 1, 1], padding='SAME')
            pre_activation = tf.nn.bias_add(conv, biases)
            conv2 = tf.nn.relu(pre_activation, name='conv2')
    
        with tf.variable_scope('pooling2_lrn') as scope:
            norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
            pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME', name='pooling2')
    
        with tf.variable_scope('local3') as scope:
            reshape = tf.reshape(pool2, shape=[self.batch_size, -1])
            dim = reshape.get_shape()[1].value
            weights = tf.Variable(tf.truncated_normal(shape=[dim, 128], stddev=0.005, dtype=tf.float32),
                                  name='weights', dtype=tf.float32)
    
            biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[128]),
                                 name='biases', dtype=tf.float32)
            local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        with tf.variable_scope('local4') as scope:
            weights = tf.Variable(tf.truncated_normal(shape=[128, 128], stddev=0.005, dtype=tf.float32),
                                  name='weights', dtype=tf.float32)
    
            biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[128]),
                                 name='biases', dtype=tf.float32)
    
            local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name='local4')
        with tf.variable_scope('softmax_linear') as scope:
            weights = tf.Variable(tf.truncated_normal(shape=[128, self.n_classes], stddev=0.005, dtype=tf.float32),
                                  name='softmax_linear', dtype=tf.float32)
    
            biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[self.n_classes]),
                                 name='biases', dtype=tf.float32)
            self.softmax_linear = tf.add(tf.matmul(local4, weights), biases, name='softmax_linear')
            
#######################################
            #
            ################################

    def structure_2(self,images):
        with tf.variable_scope('conv1') as scope:
            weights = tf.get_variable('weights',
                                 shape = [3,3,3,16],
                                 dtype = tf.float32,
                                 initializer =tf.orthogonal_initializer())
            biases = tf.get_variable('biases',
                                     shape = [16],
                                     dtype = tf.float32,
                                     initializer = tf.constant_initializer(0.1))
            conv = tf.nn.conv2d(images, weights, strides=[1,1,1,1], padding='SAME')
            pre_activation = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(pre_activation, name = scope.name)
            
        with tf.variable_scope('pooling1_lrn') as scope:
            pool1 = tf.nn.max_pool(conv1, ksize = [1,3,3,1], strides = [1,2,2,1],
                                  padding = 'SAME', name = 'pooling1')
            norm1 = tf.nn.lrn(pool1, depth_radius=4, bias = 1.0, alpha=0.001/9.0,
                             beta=0.75,name='norm1')
        with tf.variable_scope('conv2') as scope:
            weights = tf.get_variable('weights',
                                  shape=[3,3,16,16],
                                  dtype=tf.float32,
                                  initializer =tf.orthogonal_initializer())
            biases = tf.get_variable('biases',
                                    shape=[16],
                                    dtype=tf.float32,
                                    initializer=tf.constant_initializer(0.1))
            conv = tf.nn.conv2d(norm1, weights, strides=[1,1,1,1], padding = 'SAME')
            pre_activation = tf.nn.bias_add(conv, biases)
            conv2 = tf.nn.relu(pre_activation, name='conv2')
        with tf.variable_scope('pooling2_lrn') as scope:
            norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha = 0.001/9.0,
                             beta=0.75,name='norm2')
            pool2 = tf.nn.max_pool(norm2,ksize=[1,3,3,1],strides=[1,1,1,1],
                                  padding='SAME',name='pooling2')
        with tf.variable_scope('local3') as scope:
            reshape = tf.reshape(pool2,shape=[self.batch_size, -1])
            dim = reshape.get_shape()[1].value
            weights = tf.get_variable('weights',
                                     shape=[dim,128],
                                     dtype=tf.float32,
                                     initializer=tf.orthogonal_initializer())
            biases = tf.get_variable('biases',
                                    shape=[128],
                                    dtype=tf.float32,
                                    initializer=tf.constant_initializer(0.1))
            local3 = tf.nn.relu(tf.matmul(reshape, weights)+biases, name=scope.name)
        with tf.variable_scope('local4') as scope:
            weights = tf.get_variable('weights',
                                     shape=[128,128],
                                     dtype=tf.float32,
                                     initializer=tf.orthogonal_initializer())
            biases = tf.get_variable('biases',
                                    shape=[128],
                                    dtype = tf.float32,
                                    initializer = tf.constant_initializer(0.1))
            local4 = tf.nn.relu(tf.matmul(local3,weights)+biases, name='local4')
        with tf.variable_scope('softmax_layer') as scope:
            weights = tf.get_variable('softmax_linear',
                                     shape=[128,self.n_classes],
                                     dtype=tf.float32,
                                     initializer=tf.orthogonal_initializer())
            biases = tf.get_variable('biases',
                                    shape=[self.n_classes],
                                    dtype=tf.float32,
                                    initializer=tf.constant_initializer(0.1))
            self.softmax_linear=tf.add(tf.matmul(local4, weights),biases,name='softmax_linear')
        
    ################################
######
######################################

    def weight_variable(self, shape, n, use_l2, lam):
        weight = tf.Variable(tf.truncated_normal(shape, stddev=1 / n))
        # L2正则化
        if use_l2 is True:
            weight_loss = tf.multiply(tf.nn.l2_loss(weight), lam, name='weight_loss')
            tf.add_to_collection('losses', weight_loss)
        return weight

    def bias_variable(self, shape):
        bias = tf.Variable(tf.constant(0.1, shape=shape))
        return bias


    def conv2d(self, x, w):
        return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

    #learning_rate
    def structure_3(self, images):
        #image=[B,64,64,3]
        with tf.name_scope('conv1_layer'):
            w_conv1 =self.weight_variable([3, 3, 3, 64], 64, use_l2=False, lam=0)
            b_conv1 = self.bias_variable([64])
            conv_kernel1 = self.conv2d(images, w_conv1)#[B,64,64,64]
            #print("################################################################")
            #print(conv_kernel1)
            bn1 = tf.layers.batch_normalization(conv_kernel1, training=self.is_training)
            conv1 = tf.nn.relu(tf.nn.bias_add(bn1, b_conv1))
        
            w_conv2 = self.weight_variable([3, 3, 64, 64], 64, use_l2=False, lam=0)
            b_conv2 = self.bias_variable([64])
            conv_kernel2 = self.conv2d(conv1, w_conv2)#[B,64,64,64]
            bn2 = tf.layers.batch_normalization(conv_kernel2, training=self.is_training)
            conv2 = tf.nn.relu(tf.nn.bias_add(bn2, b_conv2))
        
            pool1 = self.max_pool_2x2(conv2)  #[B,32,32,64]
            #print("????????????????????????????????????????????????????????????")
            #print(pool1)
            result1 = pool1
        
        # 第二个卷积层 size:112
        # 卷积核3[3, 3, 64, 128]
        # 卷积核4[3, 3, 128, 128]
        with tf.name_scope('conv2_layer'):
            w_conv3 = self.weight_variable([3, 3, 64, 128], 128, use_l2=False, lam=0)
            b_conv3 = self.bias_variable([128])
            conv_kernel3 = self.conv2d(result1, w_conv3)#[B,32,32,128]
            bn3 = tf.layers.batch_normalization(conv_kernel3, training=self.is_training)
            conv3 = tf.nn.relu(tf.nn.bias_add(bn3, b_conv3))
        
            w_conv4 = self.weight_variable([3, 3, 128, 128], 128, use_l2=False, lam=0)
            b_conv4 = self.bias_variable([128])
            conv_kernel4 = self.conv2d(conv3, w_conv4)
            bn4 = tf.layers.batch_normalization(conv_kernel4, training=self.is_training)
            conv4 = tf.nn.relu(tf.nn.bias_add(bn4, b_conv4))#[B,32,32,128]
        
            pool2 = self.max_pool_2x2(conv4)  # 112*112 -> 56*56#[B,16,16,128]
            result2 = pool2
        
        # 第三个卷积层 size:56
        # 卷积核5[3, 3, 128, 256]
        # 卷积核6[3, 3, 256, 256]
        # 卷积核7[3, 3, 256, 256]
        with tf.name_scope('conv3_layer'):
            w_conv5 = self.weight_variable([3, 3, 128, 256], 256, use_l2=False, lam=0)
            b_conv5 = self.bias_variable([256])
            conv_kernel5 = self.conv2d(result2, w_conv5)#[B,16,16,256]
            bn5 = tf.layers.batch_normalization(conv_kernel5, training=self.is_training)
            conv5 = tf.nn.relu(tf.nn.bias_add(bn5, b_conv5))
        
            w_conv6 = self.weight_variable([3, 3, 256, 256], 256, use_l2=False, lam=0)
            b_conv6 = self.bias_variable([256])
            conv_kernel6 = self.conv2d(conv5, w_conv6)
            bn6 = tf.layers.batch_normalization(conv_kernel6, training=self.is_training)
            conv6 = tf.nn.relu(tf.nn.bias_add(bn6, b_conv6))
        
            w_conv7 = self.weight_variable([3, 3, 256, 256], 256, use_l2=False, lam=0)
            b_conv7 = self.bias_variable([256])
            conv_kernel7 = self.conv2d(conv6, w_conv7)
            bn7 = tf.layers.batch_normalization(conv_kernel7, training=self.is_training)
            conv7 = tf.nn.relu(tf.nn.bias_add(bn7, b_conv7))
        
            pool3 = self.max_pool_2x2(conv7)  # 56*56 -> 28*28
            result3 = pool3
        
        # 第四个卷积层 size:#[B,8,8,256]
        # 卷积核8[3, 3, 256, 512]
        # 卷积核9[3, 3, 512, 512]
        # 卷积核10[3, 3, 512, 512]
        with tf.name_scope('conv4_layer'):
            w_conv8 = self.weight_variable([3, 3, 256, 512], 512, use_l2=False, lam=0)
            b_conv8 = self.bias_variable([512])
            conv_kernel8 = self.conv2d(result3, w_conv8)
            bn8 = tf.layers.batch_normalization(conv_kernel8, training=self.is_training)
            conv8 = tf.nn.relu(tf.nn.bias_add(bn8, b_conv8))
        
            w_conv9 = self.weight_variable([3, 3, 512, 512], 512, use_l2=False, lam=0)
            b_conv9 = self.bias_variable([512])
            conv_kernel9 = self.conv2d(conv8, w_conv9)
            bn9 = tf.layers.batch_normalization(conv_kernel9, training=self.is_training)
            conv9 = tf.nn.relu(tf.nn.bias_add(bn9, b_conv9))
        
            w_conv10 = self.weight_variable([3, 3, 512, 512], 512, use_l2=False, lam=0)
            b_conv10 = self.bias_variable([512])
            conv_kernel10 = self.conv2d(conv9, w_conv10)
            bn10 = tf.layers.batch_normalization(conv_kernel10, training=self.is_training)
            conv10 = tf.nn.relu(tf.nn.bias_add(bn10, b_conv10))
        
            pool4 = self.max_pool_2x2(conv10)  # 28*28 -> 14*14
            result4 = pool4
        
        # 第五个卷积层 size:14#[B,4,4,512]
        # 卷积核11[3, 3, 512, 512]
        # 卷积核12[3, 3, 512, 512]
        # 卷积核13[3, 3, 512, 512]
        with tf.name_scope('conv5_layer'):
            w_conv11 = self.weight_variable([3, 3, 512, 512], 512, use_l2=False, lam=0)
            b_conv11 = self.bias_variable([512])
            conv_kernel11 = self.conv2d(result4, w_conv11)
            bn11 = tf.layers.batch_normalization(conv_kernel11, training=self.is_training)
            conv11 = tf.nn.relu(tf.nn.bias_add(bn11, b_conv11))
        
            w_conv12 = self.weight_variable([3, 3, 512, 512], 512, use_l2=False, lam=0)
            b_conv12 =self. bias_variable([512])
            conv_kernel12 = self.conv2d(conv11, w_conv12)
            bn12 = tf.layers.batch_normalization(conv_kernel12, training=self.is_training)
            conv12 = tf.nn.relu(tf.nn.bias_add(bn12, b_conv12))
        
            w_conv13 = self.weight_variable([3, 3, 512, 512], 512, use_l2=False, lam=0)
            b_conv13 = self.bias_variable([512])
            conv_kernel13 =self.conv2d(conv12, w_conv13)
            bn13 = tf.layers.batch_normalization(conv_kernel13, training=self.is_training)
            conv13 = tf.nn.relu(tf.nn.bias_add(bn13, b_conv13))
        
            pool5 = self.max_pool_2x2(conv13)  # 14*14 -> 7*7#[B,2,2,512]
            result5 = pool5
        
        # 第一个全连接层 size:7
        # 隐藏层节点数 4096
        with tf.name_scope('fc1_layer'):
            w_fc14 = self.weight_variable([2 * 2 * 512, 4096], 4096, use_l2=self.is_use_l2, lam=self.lam)
            b_fc14 =self.bias_variable([4096])
            result5_flat = tf.reshape(result5, [-1, 2 * 2 * 512])
            fc14 = tf.nn.relu(tf.nn.bias_add(tf.matmul(result5_flat, w_fc14), b_fc14))#[B,4096]
            # result6 = fc14
            result6 = tf.nn.dropout(fc14, self.keep_prob)
        
        # 第二个全连接层
        # 隐藏层节点数 4096
        with tf.name_scope('fc2_layer'):
            w_fc15 = self.weight_variable([4096, 4096], 4096, use_l2=self.is_use_l2, lam=self.lam)
            b_fc15 =self.bias_variable([4096])
            fc15 = tf.nn.relu(tf.nn.bias_add(tf.matmul(result6, w_fc15), b_fc15))
            # result7 = fc15
            result7 = tf.nn.dropout(fc15, self.keep_prob)
        
        # 输出层
        with tf.name_scope('output_layer'):
            w_fc16 = self.weight_variable([4096, self.n_classes], self.n_classes, use_l2=self.is_use_l2, lam=self.lam)
            b_fc16 = self.bias_variable([self.n_classes])
            fc16 = tf.matmul(result7, w_fc16) + b_fc16
            self.softmax_linear = tf.nn.softmax(fc16)
               

    # -----------------------------------------------------------------------------
    # loss计算
    # 传入参数：logits，网络计算输出值。labels，真实值，在这里是0或者1
    # 返回参数：loss，损失值
    def losses(self, labels):
        with tf.variable_scope('loss') as scope:
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.softmax_linear, labels=labels,
                                                                           name='xentropy_per_example')
            self.loss = tf.reduce_mean(cross_entropy, name='loss')
            tf.summary.scalar(scope.name + '/loss', self.loss)

    
    # --------------------------------------------------------------------------
    def summery(self,summary_op,step):
        summary_str = self.sess.run(summary_op)
        self.train_writer.add_summary(summary_str, step)
        
    def opt(self):
        with tf.name_scope('optimizer'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            global_step = tf.Variable(0, name='global_step', trainable=False)
            self.train_op = self.optimizer.minimize(self.loss, global_step=global_step)
        
    def train(self):
        if self.model_type==3:
            feed_dict={self.keep_prob:0.8, self.is_training:True,
                       self.is_use_l2:True,
                       self.lam:0.001}
            _, tra_loss, tra_acc = self.sess.run([self.train_op, self.loss, self.accuracy],feed_dict=feed_dict)
        else:
            _, tra_loss, tra_acc = self.sess.run([self.train_op, self.loss, self.accuracy])
        return tra_loss, tra_acc
    # -----------------------------------------------------------------------

    def evaluation(self, labels):
        with tf.variable_scope('accuracy') as scope:
            correct = tf.nn.in_top_k(self.softmax_linear, labels, 1)
            correct = tf.cast(correct, tf.float16)
            self.accuracy = tf.reduce_mean(correct)
            tf.summary.scalar(scope.name + '/accuracy',self.accuracy)
    

