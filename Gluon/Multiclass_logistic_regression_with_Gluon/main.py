import model
import mxnet as mx
#implementation

#dataset = MNIST or CIFAR10
result=model.muitlclass_logistic_regression(epoch=100, batch_size=256 , save_period=50 , load_period=100 , optimizer="sgd", learning_rate=0.1, dataset="MNIST", ctx=mx.gpu(0))
print("///"+result+"///")