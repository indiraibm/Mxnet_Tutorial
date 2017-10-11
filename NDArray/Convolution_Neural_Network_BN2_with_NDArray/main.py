import model
import mxnet as mx
#implementation

#dataset = MNIST or CIFAR10 or FashionMNIST

#if using the cpu version, You will have to wait long long time.

#when epoch=0 = testmod

result=model.CNN(epoch=20, batch_size=64 , save_period=20, load_period=20 ,  weight_decay=0 , learning_rate=0.001, dataset="CIFAR10", ctx=mx.gpu(0))
print("///"+result+"///")