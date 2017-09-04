import numpy as np
import mxnet as mx
import mxnet.gluon as gluon #when using data load
import mxnet.ndarray as nd
import mxnet.autograd as autograd
import matplotlib.pyplot as plt
from tqdm import *
import os

''' ConvolutionAutoencoder '''
def transform(data, label):
    return nd.transpose(data.astype(np.float32), (2,0,1))/255.0, label.astype(np.float32)

def CIFAR10(batch_size):

    #transform = lambda data, label: (data.astype(np.float32) / 255.0 , label)
    train_data = gluon.data.DataLoader(gluon.data.vision.CIFAR10(root="CIFAR10", train = True, transform=transform) , batch_size , shuffle=True , last_batch="rollover") #Loads data from a dataset and returns mini-batches of data.
    test_data = gluon.data.DataLoader(gluon.data.vision.CIFAR10(root="CIFAR10", train = False, transform=transform) , 10000 , shuffle=True) #Loads data from a dataset and returns mini-batches of data.

    return train_data , test_data

#evaluate the data
def generate_image(data_iterator , network , ctx ):

    for i,(data, label) in enumerate(data_iterator):

        data = data.as_in_context(ctx)
        output = network(data,0.0) # when test , 'Dropout rate' must be 0.0
        
        data = data.asnumpy()
        output=output.asnumpy()

        if i==0:
            break
    '''test'''
    column_size=10 ; row_size=10 #     column_size x row_size <= 10000

    data = data.transpose(0,2,3,1)
    output = output.transpose(0,2,3,1)

    print("show image")
    '''generator image visualization'''

    fig_g, ax_g = plt.subplots(row_size, column_size, figsize=(column_size, row_size))
    fig_g.suptitle('generator')
    for j in range(row_size):
        for i in range(column_size):
            ax_g[j][i].set_axis_off()
            ax_g[j][i].imshow(output[i + j * column_size])

    #fig_g.savefig("generator.png")
    '''real image visualization'''
    fig_r, ax_r = plt.subplots(row_size, column_size, figsize=(column_size, row_size))
    fig_r.suptitle('real')
    for j in range(row_size):
        for i in range(column_size):
            ax_r[j][i].set_axis_off()
            ax_r[j][i].imshow(data[i + j * column_size])
    fig_r.savefig("real.png")
    plt.show()


#reduce dimensions -> similar to PCA
def CNN_Autoencoder(epoch = 100 , batch_size=10, save_period=10 , load_period=100 , weight_decay=0.001 ,learning_rate= 0.1 , dataset = "CIFAR10", ctx=mx.gpu(0)):

    #data selection
    if dataset =="CIFAR10":
        train_data , test_data = CIFAR10(batch_size)
    else:
        return "The dataset does not exist."

    if dataset == "CIFAR10":
        path = "weights/CIFAR10_weights-{}".format(load_period)


    if os.path.exists(path):
        print("loading weights")
        [W1, B1, W2, B2, W3, B3, W4, B4, W5, B5, W6, B6, W7, B7, W8, B8, W9, B9, W10, B10, W11, B11, W12, B12, W13, B13, W14, B14] = nd.load(path)  # weights load

        W1=W1.as_in_context(ctx)
        B1=B1.as_in_context(ctx)
        W2=W2.as_in_context(ctx)
        B2=B2.as_in_context(ctx)
        W3=W3.as_in_context(ctx)
        B3=B3.as_in_context(ctx)
        W4=W4.as_in_context(ctx)
        B4=B4.as_in_context(ctx)
        W5=W5.as_in_context(ctx)
        B5=B5.as_in_context(ctx)
        W6=W6.as_in_context(ctx)
        B6=B6.as_in_context(ctx)

        W7=W7.as_in_context(ctx)
        B7=B7.as_in_context(ctx)

        W8=W8.as_in_context(ctx)
        B8=B8.as_in_context(ctx)
        W9=W9.as_in_context(ctx)
        B9=B9.as_in_context(ctx)
        W10=W10.as_in_context(ctx)
        B10=B10.as_in_context(ctx)
        W11=W11.as_in_context(ctx)
        B11=B11.as_in_context(ctx)
        W12=W12.as_in_context(ctx)
        B12=B12.as_in_context(ctx)
        W13=W13.as_in_context(ctx)
        B13=B13.as_in_context(ctx)
        W14=W14.as_in_context(ctx)
        B14=B14.as_in_context(ctx)

        params = [W1, B1, W2, B2, W3, B3, W4, B4, W5, B5, W6, B6, W7, B7, W8, B8, W9, B9, W10, B10, W11, B11, W12, B12, W13, B13, W14, B14]
    else:
        print("initializing weights")
        W1 = nd.random_normal(loc=0 , scale=0.1 , shape=(60,3,5,5) , ctx=ctx)
        B1 = nd.random_normal(loc=0 , scale=0.1 , shape=60 , ctx=ctx)

        W2 = nd.random_normal(loc=0 , scale=0.1 , shape=(50,60,3,3) , ctx=ctx)
        B2 = nd.random_normal(loc=0 , scale=0.1 , shape=50 , ctx=ctx)

        W3 = nd.random_normal(loc=0 , scale=0.1 , shape=(40,50,3,3) , ctx=ctx)
        B3 = nd.random_normal(loc=0 , scale=0.1 , shape=40 , ctx=ctx)

        W4 = nd.random_normal(loc=0 , scale=0.1 , shape=(30,40,3,3) , ctx=ctx)
        B4 = nd.random_normal(loc=0 , scale=0.1 , shape=30 , ctx=ctx)

        W5 = nd.random_normal(loc=0 , scale=0.1 , shape=(20,30,3,3) , ctx=ctx)
        B5 = nd.random_normal(loc=0 , scale=0.1 , shape=20 , ctx=ctx)

        W6 = nd.random_normal(loc=0 , scale=0.1 , shape=(10,20,3,3) , ctx=ctx)
        B6 = nd.random_normal(loc=0 , scale=0.1 , shape=10 , ctx=ctx)

        #####################################################################

        W7 = nd.random_normal(loc=0 , scale=0.1 , shape=(5,10,4,4) , ctx=ctx)
        B7 = nd.random_normal(loc=0 , scale=0.1 , shape=5 , ctx=ctx)

        #####################################################################

        W8 = nd.random_normal(loc=0 , scale=0.1 , shape=(10,5,3,3) , ctx=ctx)
        B8 = nd.random_normal(loc=0 , scale=0.1 , shape=10 , ctx=ctx)

        W9 = nd.random_normal(loc=0 , scale=0.1 , shape=(20,10,3,3) , ctx=ctx)
        B9 = nd.random_normal(loc=0 , scale=0.1 , shape=20, ctx=ctx)

        W10 = nd.random_normal(loc=0 , scale=0.1 , shape=(30,20,3,3) , ctx=ctx)
        B10 = nd.random_normal(loc=0 , scale=0.1 , shape=30 , ctx=ctx)

        W11 = nd.random_normal(loc=0 , scale=0.1 , shape=(40,30,3,3) , ctx=ctx)
        B11 = nd.random_normal(loc=0 , scale=0.1 , shape=40 , ctx=ctx)

        W12 = nd.random_normal(loc=0 , scale=0.1 , shape=(50,40,3,3) , ctx=ctx)
        B12 = nd.random_normal(loc=0 , scale=0.1 , shape=50 , ctx=ctx)

        W13 = nd.random_normal(loc=0 , scale=0.1 , shape=(60,50,3,3) , ctx=ctx)
        B13 = nd.random_normal(loc=0 , scale=0.1 , shape=60 , ctx=ctx)

        W14 = nd.random_normal(loc=0 , scale=0.1 , shape=(3,60,5,5) , ctx=ctx)
        B14 = nd.random_normal(loc=0 , scale=0.1 , shape=3 , ctx=ctx)

        params = [W1 , B1 , W2 , B2 , W3 , B3 , W4 , B4 , W5 , B5 , W6 , B6 , W7 , B7 , W8 , B8 , W9 , B9 , W10 , B10 , W11 , B11 , W12 , B12 , W13 , B13 , W14 , B14]

    # attach gradient!!!
    for i, param in enumerate(params):
        param.attach_grad()
    
    '''Brief description of deconvolution.
    I was embarrassed when I first heard about deconvolution,
    but it was just the opposite of convolution.
    The formula is as follows.

    The convolution formula is  output_size = ([input_size+2*pad-kernel_size]/stride) + 1

    The Deconvolution formula is output_size = stride(input_size-1)+kernel-2*pad

    '''
    def network(X,dropout=0.0):

        #encoder
        EC_H1=nd.Activation(data= nd.Convolution(data=X , weight = W1 , bias = B1 , kernel=(5,5) , stride=(1,1)  , num_filter=60) , act_type="relu") # CIFAR10 : : result = ( batch size , 60 , 28 , 28)
        EC_H2=nd.Activation(data= nd.Convolution(data=EC_H1 , weight = W2 , bias = B2 , kernel=(3,3) , stride=(1,1) , num_filter=50), act_type="relu") #  CIFAR10 : result = ( batch size , 50 , 26 , 26)
        EC_H3=nd.Activation(data= nd.Convolution(data=EC_H2 , weight = W3 , bias = B3 , kernel=(3,3) , stride=(1,1) , num_filter=40), act_type="relu") #  CIFAR10 : result = ( batch size , 40 , 24 , 24)
        EC_H4=nd.Activation(data = nd.Convolution(data=EC_H3 , weight=W4 , bias=B4 , kernel=(3,3), stride=(1,1), num_filter=30), act_type="relu")  # CIFAR10 : result = ( batch size , 30 , 22 , 22)
        EC_H5=nd.Activation(data = nd.Convolution(data=EC_H4 , weight=W5 , bias=B5 , kernel=(3,3), stride=(1,1), num_filter=20), act_type="relu")  # CIFAR10 : result = ( batch size , 20 , 20 , 20)
        EC_H6=nd.Activation(data=nd.Convolution(data=EC_H5, weight=W6, bias=B6, kernel=(3, 3), stride=(1, 1), num_filter=10),act_type="relu")  # CIFAR10 : result = ( batch size , 10 , 18 , 18)

        #Middle
        MC_H=nd.Activation(data= nd.Convolution(data=EC_H6 , weight = W7 , bias = B7 , kernel=(4,4) , stride=(2,2) , num_filter=5), act_type="relu") #  CIFAR10 : result = ( batch size , 5 , 8 , 8)

        #decoder -  why not using Deconvolution? because NDArray.Deconvolution is not working...
        DC_H1=nd.Activation(data = nd.Convolution(data=MC_H , weight = W8 , bias = B8 , kernel=(3,3) , stride=(1,1) , pad= (6,6) , num_filter=10) , act_type="relu") # CIFAR10 : : result = ( batch size , 10 , 18 , 18)
        DC_H2=nd.Activation(data= nd.Convolution(data=DC_H1 , weight = W9 , bias = B9 , kernel=(3,3) , stride=(1,1) , pad= (2,2) , num_filter=20), act_type="relu") #  CIFAR10 : result = ( batch size , 20 , 20 , 20)
        DC_H3=nd.Activation(data= nd.Convolution(data=DC_H2 , weight = W10 , bias = B10 , kernel=(3,3) , stride=(1,1) , pad= (2,2) , num_filter=30) , act_type="relu") # CIFAR10 : result = ( batch size , 30 , 22 , 22)
        DC_H4 = nd.Activation(data=nd.Convolution(data=DC_H3, weight=W11, bias=B11, kernel=(3, 3), stride=(1, 1), pad=(2, 2), num_filter=40), act_type="relu")  # CIFAR10 : result = ( batch size , 40 , 24 , 24)
        DC_H5 = nd.Activation(data=nd.Convolution(data=DC_H4, weight=W12, bias=B12, kernel=(3, 3), stride=(1, 1), pad=(2, 2), num_filter=50), act_type="relu")  # CIFAR10 : result = ( batch size , 50 , 26 , 26)
        DC_H6 = nd.Activation(data=nd.Convolution(data=DC_H5, weight=W13, bias=B13, kernel=(3, 3), stride=(1, 1), pad=(2, 2), num_filter=60), act_type="relu")  # CIFAR10 : result = ( batch size , 60 , 28 , 28)

        #output
        out=nd.Activation(data= nd.Convolution(data=DC_H6 , weight = W14 , bias = B14 , kernel=(5,5) , stride=(1,1) , pad=(4,4) , num_filter=3) , act_type="sigmoid") # CIFAR10 : : result = ( batch size , 3 , 32 , 32)

        return out

    def MSE(output, label):
        return nd.sum(0.5*nd.square(output-label) , axis=0 , exclude=True)

    #optimizer
    def SGD(params, lr , wd , bs):
        for param in params:
             param -= ((lr * param.grad)/bs+wd*param)


    for i in tqdm(range(1,epoch+1,1)):

        for data,label in train_data:
            data = data.as_in_context(ctx)
            data_ = data

            with autograd.record():
                output = network(data,0.0)

                #loss definition
                loss = MSE(output,data_) # (batch_size,)
                cost = nd.mean(loss, axis=0).asscalar()

            loss.backward()
            SGD(params, learning_rate , weight_decay , batch_size)

        print(" epoch : {} , last batch cost : {}".format(i,cost))

        #weight_save
        if i % save_period==0:

            if not os.path.exists("weights"):
                os.makedirs("weights")

            print("saving weights")
            if dataset=="CIFAR10":
                nd.save("weights/CIFAR10_weights-{}".format(i),params)

    #show image
    generate_image(test_data , network , ctx )

    return "optimization completed"

if __name__ == "__main__":
    CNN_Autoencoder(epoch=100, batch_size=128, save_period=100 , load_period=100 , weight_decay=0.001 ,learning_rate=0.1, dataset="CIFAR10", ctx=mx.gpu(0))
else :
    print("Imported")



