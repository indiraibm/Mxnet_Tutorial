import model
import mxnet as mx
#implementation

#dataset = MNIST or CIFAR10 or FashionMNIST

#if using the cpu version, You will have to wait long long time.
result=model.CNN(epoch=0, batch_size=256 , save_period=100 , load_period=100 ,  weight_decay=0.0001 , learning_rate=0.1, dataset="MNIST", ctx=mx.gpu(0))
print("///"+result+"///")