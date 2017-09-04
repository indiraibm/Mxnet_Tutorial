import model
import mxnet as mx
#implementation

#dataset = MNIST or CIFAR10
result=model.CNN_Autoencoder(epoch=1, batch_size=128 , save_period=1000 , load_period=100 ,  weight_decay=0.0001 , learning_rate=0.001, dataset="CIFAR10", ctx=mx.gpu(0))
print("///"+result+"///")
