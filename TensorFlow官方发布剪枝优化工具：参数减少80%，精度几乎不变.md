## TensorFlow官方发布剪枝优化工具：参数减少80%，精度几乎不变

关注前沿科技 [量子位](javascript:void(0);) *今天*

##### 晓查 编译自 Medium 量子位 报道 | 公众号 QbitAI

![img](https://mmbiz.qpic.cn/mmbiz_gif/YicUhk5aAGtCibAUvicJXRlt3YnSy98cUFknaF3oItQicpv7rB57xgcz6NIMJvPQLy4bmXryu7ZsHqPModd6Qicibnqw/640?wx_fmt=gif&tp=webp&wxfrom=5&wx_lazy=1)

去年TensorFlow官方推出了模型优化工具，最多能将模型尺寸减小4倍，运行速度提高3倍。

最近现又有一款新工具加入模型优化“豪华套餐”，这就是基于Keras的**剪枝优化**工具。

训练AI模型有时需要大量硬件资源，但不是每个人都有4个GPU的豪华配置，剪枝优化可以帮你缩小模型尺寸，以较小的代价进行推理。

## 什么是权重剪枝？

权重剪枝（Weight Pruning）优化，就是消除权重张量中不必要的值，减少神经网络层之间的连接数量，减少计算中涉及的参数，从而降低操作次数。

![img](https://mmbiz.qpic.cn/mmbiz_png/YicUhk5aAGtCibAUvicJXRlt3YnSy98cUFkywz7arHLfsKicNvJ7R228jXhx3YdMZjdOm2iaeN9tVfrNSqtLPNXrvQw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

这样做的好处是压缩了网络的存储空间，尤其是稀疏张量特别适合压缩。例如，经过处理可以将MNIST的90％稀疏度模型从12MB压缩到2MB。

此外，权重剪枝与量化（quantization）兼容，从而产生复合效益。通过训练后量化（post-training quantization），还能将剪枝后的模型从2MB进一步压缩到仅0.5MB 。

TensorFlow官方承诺，将来TensorFlow Lite会增加对稀疏表示和计算的支持，从而扩展运行内存的压缩优势，并释放性能提升。

## 优化效果

权重剪枝优化可以用于不同任务、不同类型的模型，从图像处理的CNN用于语音处理的RNN。下表显示了其中一些实验结果。

![img](https://mmbiz.qpic.cn/mmbiz_png/YicUhk5aAGtCibAUvicJXRlt3YnSy98cUFksxr3xtTKjccZsQJGN2D0ia3fHWWLwU09YPFPKeNVvfNp3NVDiaDqHrIw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

以GNMT从德语翻译到英语的模型为例，原模型的BLEU为29.47。指定80%的稀疏度，经优化后，张量中的非零参数可以从211M压缩到44M，准确度基本没有损失。

## 使用方法

现在的权重剪枝API建立在Keras之上，因此开发者可以非常方便地将此技术应用于任何现有的Keras训练模型中。

开发者可以指定最终目标稀疏度（比如50%），以及执行剪枝的计划（比如2000步开始剪枝，在4000步时停止，并且每100步进行一次)，以及剪枝结构的可选配置。

```
import tensorflow_model_optimization as tfmot

model = build_your_model()

pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
initial_sparsity=0.0, final_sparsity=0.5,
begin_step=2000, end_step=4000)

model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(model, pruning_schedule=pruning_schedule)

…

model_for_pruning.fit(…)
```



![img](https://mmbiz.qpic.cn/mmbiz_png/YicUhk5aAGtCibAUvicJXRlt3YnSy98cUFkPZ1oaOnhv1mjuXGT0ZgMZBXGsbhoht8UPJ39YiaeiaHeI9l9rjEP1fXQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

###### **△** 三个不同张量，左边的没有稀疏度，中心的有多个单独0值，右边的有1x2的稀疏块。

随着训练的进行，剪枝过程开始被执行。在这个过程中，它会消除消除张量中最接近零的权重，直到达到当前稀疏度目标。

每次计划执行剪枝程序时，都会重新计算当前稀疏度目标，根据平滑上升函数逐渐增加稀疏度来达到最终目标稀疏度，从0%开始直到结束。

![img](https://mmbiz.qpic.cn/mmbiz_png/YicUhk5aAGtCibAUvicJXRlt3YnSy98cUFkyT5KyytoiasjbtO6spNMiaBEbBqv9Ujia3StO84TzHqKYjgAv9OUNj7oQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

用户也可以根据需要调整这个上升函数。在某些情况下，可以安排训练过程在某个步骤达到一定收敛级别之后才开始优化，或者在训练总步数之前结束剪枝，以便在达到最终目标稀疏度时进一步微调系统。

![img](https://mmbiz.qpic.cn/mmbiz_gif/YicUhk5aAGtCibAUvicJXRlt3YnSy98cUFkibibCb1rU72pPicib3gdXNYNIFoOPMJTww8ykhic3jMry0hZCTyKKGfE7JA/640?wx_fmt=gif&tp=webp&wxfrom=5&wx_lazy=1)

###### **△**权重张量剪枝动画，黑色的点表示非零权重，随着训练的进行，稀疏度逐渐增加

GitHub地址：

https://github.com/tensorflow/model-optimization

官方教程：
https://www.tensorflow.org/model_optimization/guide/pruning/pruning_with_keras

— **完** —

**订阅AI内参，获取AI行业资讯**

![img](https://mmbiz.qpic.cn/mmbiz_jpg/YicUhk5aAGtAkpibldb6tu0lfWoPMdPlFKOhiaKOf4PibMlFibooQe4JdMLqxAN1PpoaQfD0RfpkkSzZsEeBzR1FLwA/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

**加入社群**

量子位AI社群开始招募啦，量子位社群分：AI讨论群、AI+行业群、AI技术群；



欢迎对AI感兴趣的同学，在量子位公众号（QbitAI）对话界面回复关键字“微信群”，获取入群方式。（技术群与AI+行业群需经过审核，审核较严，敬请谅解）

**诚挚招聘**

量子位正在招募编辑/记者，工作地点在北京中关村。期待有才气、有热情的同学加入我们！相关细节，请在量子位公众号(QbitAI)对话界面，回复“招聘”两个字。

![img](https://mmbiz.qpic.cn/mmbiz_jpg/YicUhk5aAGtAWR7KMTicXl4micou1JFQYicKuicoLdfTGicbTQVODlcKpQOobfgv8PhpRbsDdXvXUia2CJZxC2tQzQzwg/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)



**量子位** QbitAI · 头条号签约作者





վ'ᴗ' ի 追踪AI技术和产品新动态



喜欢就点「好看」吧 !











微信扫一扫
关注该公众号