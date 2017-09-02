import model
import mxnet as mx
#implementation

#dataset = MNIST or CIFAR10
result=model.CNN_Autoencoder(epoch=100, batch_size=100 , save_period=100 , load_period=100 ,  weight_decay=0.00001 , learning_rate=0.1, dataset="CIFAR10", ctx=mx.gpu(0))
print("///"+result+"///")