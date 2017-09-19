import model
import mxnet as mx
#implementation

#dataset = MNIST or FashionMNIST
result=model.LottoNet(epoch = 30000 , batch_size=50, save_period=30000 , load_period=30000 ,optimizer="adam",learning_rate= 0.0009 , ctx=mx.gpu(0))
print("///"+result+"///")