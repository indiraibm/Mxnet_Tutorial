import model
import mxnet as mx
#implementation

#dataset = MNIST or CIFAR10 or FashionMNIST

#if using the cpu version, You will have to wait long long time.

#when epoch=0 = testmod

result=model.CNN(epoch=1, batch_size=256 , save_period=30, load_period=30 ,  weight_decay=0.0001 , learning_rate=0.001, dataset="MNIST", ctx=mx.gpu(0))
print("///"+result+"///")