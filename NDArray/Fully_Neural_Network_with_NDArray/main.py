import model
import mxnet as mx
#implementation

#dataset = MNIST or CIFAR10
result=model.FNN(epoch=20, batch_size=256 , save_period=20 , load_period=20 ,  weight_decay=0.0001 , learning_rate=0.1, dataset="CIFAR10", ctx=mx.gpu(0))
print("///"+result+"///")