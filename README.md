# python3
整理的三个深度学习算法。
Here are the explanations of the three algorithms
===============================================================
Preface
从input_data中我们可以得到输入图像的数据类型为shape=[B,64,64,3]，B为batch_size=20.
通过修改config模块里的"num"可以选择不同算法                      n_classes=2
我们先定义class为总类，根据算法及代码主体设置属性（self）
''内为存储名称
=========================================
第一个模型：structure_1
1,以下数据储存在卷积层名称conv1中
  生成随机变量'weights'（下称w,float32，下同）,shape=[3,3,3,64],步长为1.0；
  生成随机变量'biases'(下称b,float32,下同）,shape=[64],b=0.1;
  生成第一个卷积层conv，输入图像数据（images），w，填充方式为SAME;[20,64,64,64]
  激活：pa=conv+b；
  用激励函数relu激活pa，生成第二个卷积层
  
2，以下数据储存在pooling1_lrn中
  生成池化层'pooling1',大小为[1,3,3,1],填充方式为SAME
  生成常量'norm1'
3，以下数据储存在conv2中
  重复1和2，but w.shape=[3,3,64,16];名称标号+1;[20,32,32,64]
                                              [20,32,32,16]
4，以下数据储存在local3中
  定义重新排序池化层'pool2'，shape=[self.B,-1]--reshape=(20,16384)
  得到dim=16384,w,w.shape=[16384,128]
  x=matmul(reshape,w)       [20,128]
  定义并重复标准local3=relun（wx+b）
5，以下数据储存在local4中
  定义w,w.shape=[20，128];定义b;
  x=matmul(local3,w)；[128,128]
  
  定义并重复标准local4=relu（wx+b）
6，以下数据储存在softmax_linear中
  定义w,w.shape=[128,2];b;
  x=matmul(local4,w)
  'softmax_linear'=add(x,b)
算法一shape主要流程结束
=========================================================
第二个模型：structure_2————参考模型一
=========================================================
第三个模型：structure_3
1，'conv1_layer'
  w,w.shape=[3,3,3,64];b,b.shape=[64];
  conv1=conv2d(images,w)     [20,64,64,64]
  bn1；c1=relu(add(bn1,b))
  w,w.shape=[3,3,64,64];b;
  conv2=conv2d(conv1,w)      [20,64,64,64]
  bn2;c2=relu(add(bn2,b))
  pool1=self.max_pool_2x2(c2)   [20,32,32,64]
2，3，4，5,'conv_layer'
  同1；                    [20,2,2,512]
6，w,w.shape=[2*2*512,4096];b;p5f=(20,2048)
  a=matmul(p5f,w);fc14=relu(add(a,b)) 
  fc14=relu(add(a,b))      
  result6=dropout(fc14,self.keep_prob)  [20,4096]
7,fc2_layer
  w,w.shape=[4096,4096];b;
  同6；                    [20,4096]
8,output_layer
  w,w.shape=p4096,self.n_classes0;b;
  fc16同上
  self.softmax_linear = tf.nn.softmax(fc16)
  模型三shape主要流程结束



















