练手实验
---
这次主要是动手写了一下tensorflow，在官方手写数字识别的俩个模型上，又添加了一层网络，具体来说，就是搭建了一个这样的网络
![](http://img.blog.csdn.net/20171021021736733?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMTcyODcyOA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
就是简单将训练好的模型一和模型二的输出作为综合模型的输入，相当于一个专家评估系统，最终，再由联合模型输出最后的预测值。
实验的结果表明，这是一个很差的模型，通过几次超参数的修改测试，最高的准确率只能达到0.6。不过，作为自己写的第一模型，感觉还是挺愉快的