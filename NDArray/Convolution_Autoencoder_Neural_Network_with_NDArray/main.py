import model
import mxnet as mx
#implementation

#dataset = FashionMNIST
result=model.CNN_Autoencoder(epoch=0, batch_size=64 , save_period=100, load_period=100 ,  weight_decay=0.0 , learning_rate=0.001, dataset="FashionMNIST", ctx=mx.cpu(0))
print("///"+result+"///")
