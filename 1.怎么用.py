from tensorboardX import SummaryWriter
#啥都先建立一个SummaryWriter
'''
一般来讲，我们对于每次实验新建一个路径不同的 SummaryWriter，也叫一个 run，如 runs/exp1、runs/exp2。
接下来，我们就可以调用 SummaryWriter 实例的各种 add_something 方法向日志中写入不同类型的数据了。想要在浏览器中查看可视化这些数据，只要在命令行中开启 tensorboard 即可：

'''
writer = SummaryWriter("./visonlog/exp1")#前提这个路径要存在

# writer.add_scalar()#关于数的
'''Add scalar data to summary.'''
for  i in range(100):
    writer.add_scalar("y=x2",i*i,i)
# writer.add_image()#关于图的
writer.close()

#在中毒那输入tensorboard --logdir=你的那个SummaryWriter的路径