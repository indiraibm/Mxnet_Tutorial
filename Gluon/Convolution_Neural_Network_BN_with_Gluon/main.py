import model
import mxnet as mx
#implementation

#dataset = MNIST or CIFAR10 or FashionMNIST
result=model.CNN(epoch = 50, batch_size=128, save_period=50 , load_period=50 ,optimizer="adam",learning_rate= 0.001 , dataset = "CIFAR10", ctx=mx.gpu(0))
print("///"+result+"///")