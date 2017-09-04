import model
import mxnet as mx
#implementation

#dataset = MNIST or CIFAR10 or FashionMNIST
result=model.FNN(epoch=100, batch_size=256 , save_period=100 , load_period=100 ,  weight_decay=0.0001 , learning_rate=0.1, dataset="FashionMNIST", ctx=mx.gpu(0))
print("///"+result+"///")