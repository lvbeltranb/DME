from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM, Flatten, Embedding,Input, Multiply #Merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
#from keras.utils import plot_model
from keras.models import Model
from keras import backend
from keras.applications import vgg19, resnet50, resnet
from keras.applications import densenet
from keras.metrics import categorical_accuracy

import h5py
import pickle

def compute_embeddings_pretrained(parameters,train_X,val_X,test=None):
    if parameters['cnn'] == 'vgg':
        cnn_model = vgg19.VGG19(weights='imagenet',pooling='avg') 
        cnn = Model(inputs=cnn_model.input, outputs=cnn_model.get_layer('fc2').output)   
        
    if parameters['cnn'] == 'resnet':
        #cnn = resnet50.ResNet50(weights='imagenet',include_top=False,pooling=parameters['pool']) 
        #cnn = Model(inputs=cnn_model.input, outputs=cnn_model.get_layer('').output)  
        cnn = resnet.ResNet101(include_top=False, weights='imagenet', pooling='avg')
        
    if parameters['cnn'] == 'inc_resnet':
        cnn = inception_resnet_v2.InceptionResNetV2(weights='imagenet',include_top=False,pooling=parameters['pool']) 
        #cnn = Model(inputs=cnn_model.input, outputs=cnn_model.get_layer('').output)  
        
    if parameters['cnn'] == 'densenet':
        cnn = densenet.DenseNet121(weights='imagenet',include_top=False,pooling=parameters['pool']) 
        
    if parameters['cnn'] == 'xception':
        cnn = xception.Xception(weights='imagenet',include_top=False,pooling=parameters['pool'])
        
    '''
    if parameters['freeze']:
        for layer in cnn.layers:
            layer.trainable = False  
        print ('TRAINABLE LAYERS: ')
        for layer in cnn.layers:
            print(layer, layer.trainable)
    '''   
    embeddings_train = cnn.predict(train_X)
    embeddings_val = cnn.predict(val_X)
    print('Dim embeddings train: %s', str(embeddings_train.shape))
    print('Dim embeddings val: %s', str(embeddings_val.shape)) 
    if test is not None:
        embeddings_test = {}
        for k,v in test.items():
            if parameters['cnn'] == 'resnet': 
                embeddings_test[k] = cnn.predict(v)#.reshape((2048,-1))
            elif parameters['cnn'] == 'inc_resnet': 
                embeddings_test[k] = cnn.predict(v)#.reshape((1536,-1))
            elif parameters['cnn'] == 'xception': 
                embeddings_test[k] = cnn.predict(v)#.reshape((,-1))
            else:
                embeddings_test[k] = cnn.predict(v)#.reshape((1024,-1))
        with open(parameters['embeddings_pretrained_test'], 'wb') as f:
            pickle.dump(embeddings_test, f, protocol=pickle.HIGHEST_PROTOCOL)
        print('Dim embeddings test: %s', str(len(embeddings_test)))
    return embeddings_train.squeeze(), embeddings_val.squeeze()

def multimodal_net(embedding_matrix,parameters):
    backend.clear_session()
    image_model = visual_net(parameters)
    text_model = text_net(embedding_matrix, parameters)
    print ("Merging final model...")
    
    fc_model = Multiply()([image_model.output,text_model.output])
    
    #fc_model = Sequential()
    #fc_model.add(merged)#, mode='mul')) # keras 2.0
    
    for layer in parameters['layers']:
        fc_model = Dropout(parameters['dropout_rate'])(fc_model)
        fc_model = Dense(layer[0], activation=layer[1])(fc_model) 
        
    fc_model =  Model([image_model.input,text_model.input], fc_model)
    fc_model.compile(optimizer=parameters['optimizer'], loss=parameters['loss'],metrics=[parameters['acc']])
        
    return fc_model

# if parameters['simple']= True: model without layers after merge, last layer of lstm must be the size of the output required (LSTM)
# if parameters['simple']= False:
#     if parameters['layers']=='': model takes the last layer of model, so, only one layer extra as a classifier (LSTM+ last classification layer)
#     if parameters['layers']==[(dim,act)]: model adds one hidden layer after merge of model, and the last classifier layer as well (LSTM+hidd_layers+classification layer)
def text_net(embedding_matrix,parameters):
    print("Creating text model...")
    model = Sequential()
    model.add(Embedding(parameters['num_words'], parameters['embedding_dim'], weights=[embedding_matrix],
                        input_length=parameters['seq_length'],trainable=False))
    
    model.add(LSTM(units=512, return_sequences=True, input_shape=(parameters['seq_length'], parameters['embedding_dim'])))
    model.add(Dropout(parameters['dropout_rate']))
    model.add(LSTM(units=512, return_sequences=False))

    for layer in parameters['layers_tx']:
        model.add(Dropout(parameters['dropout_rate']))
        model.add(Dense(layer[0], activation=layer[1])) #layer(0)-->dim,layer(1)-->activation        

    return model

def visual_net(parameters):
    #backend.clear_session()
    model = Sequential()
    #1024, 4096,tanh
    model.add(Dense(parameters['layers_im'][0][0],input_dim=parameters['dim_input_cnn'],activation=parameters['layers_im'][0][1]))

    for layer in parameters['layers_im'][1:]:
        model.add(Dropout(parameters['dropout_rate']))
        model.add(Dense(layer[0], activation=layer[1])) #layer(0)-->dim,layer(1)-->activation   
  
    return model
