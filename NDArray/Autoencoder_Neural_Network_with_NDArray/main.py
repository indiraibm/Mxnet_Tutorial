import model
import mxnet as mx
#implementation

#dataset = MNIST or FashionMNIST
result=model.Autoencoder(epoch=100, batch_size=128 , save_period=100 , load_period=100 ,  weight_decay=0.00001 , learning_rate=0.3, dataset="MNIST", ctx=mx.gpu(0))
print("///"+result+"///")