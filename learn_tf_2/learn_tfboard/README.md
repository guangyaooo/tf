#Tensorflow可视化

tensorflow官方提供了一个强大的可视化工具——tensorboard，可以直观的显示出网络在运行过程中各个输入、输出、损失值等的变化，同时，也能显示出网络的结构，tensorflow内置的summary下的系列方法允许我们灵活的控制各项显示，本次，通过官方实例	[mnist_with_summaries.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist_with_summaries.py)对可视化操作做一个总结
概括的讲，tensorflow为开发者们提供了如下几个summary ops
* tf.summary.scalar
* tf.summary.image
* tf.summary.audio
* tf.summary.text
* tf.summary.histogram  

详细的api请参见[summary_operation](http://www.tensorfly.cn/tfdoc/api_docs/python/train.html#summary_operation)

##模型可视化比较
在创建模型时，我们经常需要调整某些超参数以使模型达到最佳的效果，因此改变了某个超参数后，我们想要看到改变之前和改变之后的模型的变化时，
就可以这样做：在改变超参数之前和改变超参数之后，我们分别将俩种情况下生成的`.tfevents`文件保存于不同的目录下，例如
```
/log_dir/test/events.out.tfevents.1508592516.DESKTOP-7607VEF
/log_dir/train/events.out.tfevents.1508592517.DESKTOP-7607VEF
```
示例代码[comparing_example.py]()

##可视化
####Scalar面板
Scalar中体重标量的可视化信息，在Scalar面板下，横轴一般为时间相关的量（比如迭代步数），纵轴为标量的数值大小  

####Histogram面板
Histogram提供变量的直方图统计信息，在该面板下，提供两种显示模式:`OVERLAY`、`OFFSET`  

#####OFFSET  

在OFFSET模式下，图形可以分为多个层，其中，每一层代表一次迭代步骤，越靠后（深颜色）的图层迭代步骤越早。此时，下图所显示的三个数字代表
着在第1725次迭代时，prediction中大概有9.69个值为0.235的元素
![histogram](http://img.blog.csdn.net/20171021225731981?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMTcyODcyOA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)  

- y轴：迭代步骤
- x轴：数值
#####OVERLAY
下图是一个在OVERLAY模式下显示的统计图，三个数字的意思是在第15次迭代时，prediction中大概有6.64个值为0.303的元素  

![这里写图片描述](http://img.blog.csdn.net/20171021231256820?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMTcyODcyOA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)  

- y轴：数量
- x轴：数值
####Distribution面板
该面板以另一种方式给出了`tf.summary_histogram`的统计信息，每一条线代表统计值中中前百分之多少的变量的变化，例如，最下面一条线代表了
最小值的变化，这些线由上到下，所带表的意义依次为：`[maximum, 93%, 84%, 69%, 50%, 31%, 16%, 7%, minimum]`
####Image面板
该面板显示了由`tf.summary_image`统计到信息，以图片的形式给出
####Audio面板
该面板可以显示音频信息
####Graph
结合`tf.name_scope`使用，可以显示出模型的结构，用来检查错误
####Exmbedding Projector
可以可视化高维度的数据，具体的使用请见[the embedding projector tutorial](https://www.tensorflow.org/get_started/embedding_viz)
####Text面板
显示一些文本信息，支持markdown语法，通过调用`tf.summary_text`来保存信息