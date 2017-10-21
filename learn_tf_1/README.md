Tensorflow总结
----

	本周学习了一个谷歌开源的机器学习框架-tensorflow，感觉这一周的学习只能算是初窥门径，很多东西还需要学习，总结一下这周对tensorflow的理解
	Tensorflow编程给我的感觉就像是在绘制一张计算图，构建模型的第一步先要对自己的模型的一个结够有清晰的了解，接下就是用tensorflow内置的各种方法生成图中的一个一个的op(operation)，计算图就是有这些ops按一定顺序执行构成的图。根据官方给的手写数字识别样例，我觉得在编写有监督学习程序时可以按照以下结构一步一步编写

- 构造inference(input,*args)函
	在inference函数内部，我们需要搭建神经网络的具体结够，inference函数接收一个input参数，作为神经网络的输入值，返回一个output，作为网络的输出值，输入层至输出层直接的结构在inference内部构建
- 构造loss(prediction,y,*args)函数
在loss函数内部，我们需要构建的是该模型的损失函数，输入值predication是interface的输出，y是所有训练样本真实值所购成的张量，loss函数返回一个损失函数的op
- 构造training(loss,learning_rate,*args)函数
	在training函数内部，我们需要定义模型参数更新的方式，确定使用哪种梯度更新方式，函数的输入loss是loss函数的返回值，learning_rate学习率，函数返回一个train_op表示一次训练操作
- 构建evaluation(prediction,y,*args)函数
	在evaluation函数内部，我们需要对模型输出进行评估，返回评估的结果
- 构建main()函数
	在main函数内，我们需要将以上的模块连接起来，构成一张完整的计算图，训练好之后运行evaluation函数，对模型进行评估