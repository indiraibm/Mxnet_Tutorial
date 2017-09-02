import numpy as np
import mxnet as mx
import mxnet.gluon as gluon
import mxnet.ndarray as nd
import mxnet.autograd as autograd
import matplotlib.pyplot as plt
from tqdm import *
import os

''' Autoencoder '''
def transform(data, label):
    return data.astype(np.float32)/255, label.astype(np.float32)

def CIFAR10(batch_size):

    #transform = lambda data, label: (data.astype(np.float32) / 255.0 , label)
    train_data = gluon.data.DataLoader(gluon.data.vision.CIFAR10(root="CIFAR10", train = True, transform=transform) , batch_size , shuffle=True , last_batch="rollover") #Loads data from a dataset and returns mini-batches of data.
    test_data = gluon.data.DataLoader(gluon.data.vision.CIFAR10(root="CIFAR10", train = False, transform=transform) , 10000 , shuffle=False) #Loads data from a dataset and returns mini-batches of data.

    return train_data , test_data

#evaluate the data
def generate_image(data_iterator , num_inputs , network , ctx ):

    for data, label in data_iterator:

        data=nd.transpose(data=data , axes=(0,3,1,2))
        output = network(data,0.0) # when test , 'Dropout rate' must be 0.0
        
        data = data.asnumpy() * 255.0
        output=output.asnumpy() * 255.0

        '''test'''
        column_size=10 ; row_size=10 #     column_size x row_size <= 10000

        '''generator image visualization'''
        fig_g, ax_g = plt.subplots(row_size, column_size, figsize=(column_size, row_size))
        fig_g.suptitle('generator')
        for j in range(row_size):
            for i in range(column_size):
                ax_g[j][i].set_axis_off()
                ax_g[j][i].imshow(np.reshape(output[i + j * column_size], (28, 28 ,3)), cmap='gray')

        fig_g.savefig("generator.png")
        '''real image visualization'''
        fig_r, ax_r = plt.subplots(row_size, column_size, figsize=(column_size, row_size))
        fig_r.suptitle('real')
        for j in range(row_size):
            for i in range(column_size):
                ax_r[j][i].set_axis_off()
                ax_r[j][i].imshow(np.reshape(data[i + j * column_size],(28 , 28 , 3)), cmap='gray')
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
        [W1 , B1 , W2 , B2 , W3 , B3 , W4 , B4 , W5 , B5 , W6 , B6 , W7 , B7 , W8 , B8] = nd.load(path)  # weights load

        W1=W1.as_in_context(ctx)
        B1=B1.as_in_context(ctx)
        W2=W2.as_in_context(ctx)
        B2=B2.as_in_context(ctx)
        W3=W3.as_in_context(ctx)
        B3=B3.as_in_context(ctx)
        W4=W4.as_in_context(ctx)
        B4=B4.as_in_context(ctx)

        params = [W1 , B1 , W2 , B2 , W3 , B3 , W4 , B4 , W5 , B5 , W6 , B6 , W7 , B7 , W8 , B8]
    else:
        print("initializing weights")
        W1 = nd.random_normal(loc=0 , scale=0.1 , shape=(40,3,7,7) , ctx=ctx)
        B1 = nd.random_normal(loc=0 , scale=0.1 , shape=40 , ctx=ctx)

        W2 = nd.random_normal(loc=0 , scale=0.1 , shape=(20,40,7,7) , ctx=ctx)
        B2 = nd.random_normal(loc=0 , scale=0.1 , shape=20 , ctx=ctx)

        W3 = nd.random_normal(loc=0 , scale=0.1 , shape=(10,20,7,7) , ctx=ctx)
        B3 = nd.random_normal(loc=0 , scale=0.1 , shape=10 , ctx=ctx)

        W4 = nd.random_normal(loc=0 , scale=0.1 , shape=(5,10,7,7) , ctx=ctx)
        B4 = nd.random_normal(loc=0 , scale=0.1 , shape=5 , ctx=ctx)
        
        #####################################################################

        W5 = nd.random_normal(loc=0 , scale=0.1 , shape=(5,10,7,7) , ctx=ctx)
        B5 = nd.random_normal(loc=0 , scale=0.1 , shape=5 , ctx=ctx)

        W6 = nd.random_normal(loc=0 , scale=0.1 , shape=(5,10,7,7) , ctx=ctx)
        B6 = nd.random_normal(loc=0 , scale=0.1 , shape=5 , ctx=ctx)

        W7 = nd.random_normal(loc=0 , scale=0.1 , shape=(5,10,7,7) , ctx=ctx)
        B7 = nd.random_normal(loc=0 , scale=0.1 , shape=5 , ctx=ctx)

        W8 = nd.random_normal(loc=0 , scale=0.1 , shape=(5,10,7,7) , ctx=ctx)
        B8 = nd.random_normal(loc=0 , scale=0.1 , shape=5 , ctx=ctx)

        params = [W1 , B1 , W2 , B2 , W3 , B3 , W4 , B4 , W5 , B5 , W6 , B6 , W7 , B7 , W8 , B8]

    # attach gradient!!!
    for i, param in enumerate(params):
        param.attach_grad()
    
    #Fully Neural Network with 1 Hidden layer 
    def network(X,dropout=0.0): # formula : output_size=((input−weights+2*Padding)/Stride)+1



        #encoder
        EC_H1=nd.Activation(data= nd.Convolution(data=X , weight = W1 , bias = B1 , kernel=(7,7) , stride=(1,1)  , num_filter=40) , act_type="relu") # CIFAR10 : : result = ( batch size , 40 , 26 , 26) 
        EC_H2=nd.Activation(data= nd.Convolution(data=EP_H1 , weight = W2 , bias = B2 , kernel=(7,7) , stride=(1,1) , num_filter=20), act_type="relu") #  CIFAR10 : result = ( batch size , 20 , 20 , 20)
        EC_H3=nd.Activation(data= nd.Convolution(data=EP_H2 , weight = W3 , bias = B3 , kernel=(7,7) , stride=(1,1) , num_filter=10), act_type="relu") #  CIFAR10 : result = ( batch size , 10 , 14 , 14)
        
        #Middle
        MC_H=nd.Activation(data= nd.Convolution(data=EP_H3 , weight = W4 , bias = B4 , kernel=(7,7) , stride=(1,1) , num_filter=5), act_type="relu") #  CIFAR10 : result = ( batch size , 5 , 8 , 8)

        #decoder
        DC_H1=nd.Activation(data= nd.Deconvolution(data=MC_H , weight = W5 , bias = B5 , kernel=(7,7) , stride=(1,1)  , num_filter=10) , act_type="relu") # CIFAR10 : : result = ( batch size , 10 , 14 , 14) 
        DC_H2=nd.Activation(data= nd.Deconvolution(data=DC_H1 , weight = W6 , bias = B6 , kernel=(7,7) , stride=(1,1) , num_filter=20), act_type="relu") #  CIFAR10 : result = ( batch size , 20 , 20 , 20)
        DC_H3=nd.Activation(data= nd.Deconvolution(data=DC_H2 , weight = W7 , bias = B7 , kernel=(7,7) , stride=(1,1)  , num_filter=40) , act_type="relu") # CIFAR10 : : result = ( batch size , 40 , 26 , 26)

        #output
        out=nd.Activation(data= nd.Deconvolution(data=DC_H3 , weight = W8 , bias = B8 , kernel=(7,7) , stride=(1,1)  , num_filter=3) , act_type="sigmoid") # CIFAR10 : : result = ( batch size , 3 , 32 , 32)       

        return out

    def MSE(output, label):
        return nd.sum(0.5*nd.square(output-label),axis=1)

    #optimizer
    def SGD(params, lr , wd , bs):
        for param in params:
             param -= ((lr * param.grad)/bs+wd*param)


    for i in tqdm(range(1,epoch+1,1)):

        for data,label in train_data:
            data=nd.transpose(data=data , axes=(0,3,1,2))
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
    generate_image(test_data , num_inputs , network , ctx )

    return "optimization completed"

if __name__ == "__main__":
    CNN_Autoencoder(epoch=100, batch_size=128, save_period=10 , load_period=100 , weight_decay=0.001 ,learning_rate=0.1, dataset="CIFAR10", ctx=mx.gpu(0))
else :
    print("Imported")


