import os, argparse, pandas as pd, sys, pickle, time, random, json, cv2, numpy as np, h5py, logging
from sklearn.metrics import accuracy_score as acc
from PIL import Image
from sklearn import svm
from sklearn.externals import joblib
from sklearn.metrics import average_precision_score
from sklearn import preprocessing
import scipy.stats as sp
from scipy.stats.stats import pearsonr   
from scipy.spatial.distance import cosine as cs
from scipy.spatial.distance import euclidean as eu
from sklearn.preprocessing import binarize
import tensorflow as tf
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from keras.utils.np_utils import to_categorical
from keras.models import Model
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.callbacks import Callback
from keras import backend as K
from keras.metrics import categorical_accuracy

src_path = 'path'
# add src_path if it's located in another path
#sys.path.append(os.path.join(src_path,'multimodal_retrieval/'))
from models.two_nets_g import *

# General function to compute average precision for retrieval given other samples as inputs
def get_AP(sorted_scores, given_sample, GT_samples, top_k):
        consider_top = sorted_scores[:top_k]
        top_sample_classes = [GT_samples[i[0]] for i in consider_top]
        class_of_sample = GT_samples[given_sample]
        T = top_sample_classes.count(class_of_sample)
        R = top_k
        sum_term = 0
        for i in range(0,R):
                if top_sample_classes[i] != class_of_sample:
                        pass
                else:
                        p_r = top_sample_classes[:i+1].count(class_of_sample)
                        sum_term = sum_term + (p_r*1.0)/len(top_sample_classes[:i+1])
        if T == 0:
                return 0
        else:
                return float(sum_term/T)
            

def calc_distances(dic1,dic2,task,topks,distances,GT_samples):
    logger = logging.getLogger("my_logger") 
    logger.info('TASK: %s',task)
    maps =[]
    for dis in distances:
        logger.info('distance: %s',dis)
        tops_val = []
        for topk in topks:
            logger.info('K: %s',topk)         
            mAP = 0
            order_of_samples1 = sorted(dic1.keys())
            order_of_samples2 = sorted(dic2.keys())
            for sample in order_of_samples1:
                score_samples = []
                reps1 = dic1[sample]
                for given_sample in order_of_samples2:
                        if sample == given_sample:
                            pass
                        else:
                            reps2 = dic2[given_sample]
                            if dis == 'cos':                
                                given_score = 1 - cs(reps1, reps2)
                            elif dis == 'sp':
                                given_score = sp.entropy(reps1, reps2)
                            elif dis == 'corr':
                                given_score = pearsonr(reps1,reps2)[0] #1-->correlated, 0-->no correlated
                            else:
                                given_score = eu(reps1, reps2)
                            score_samples.append((given_sample, given_score))
                if dis == 'corr':
                    sorted_scores = sorted(score_samples, key=lambda x:x[1],reverse=True) #decreasing                    
                else:
                    sorted_scores = sorted(score_samples, key=lambda x:x[1],reverse=False) #increasing
                mAP = mAP + get_AP(sorted_scores, sample, GT_samples, top_k=topk)

            tops_val.append(mAP/(len(dic1.keys())*1.0))
            logger.info('mAP %s %s',task, str(tops_val[-1]))
        maps.append(tops_val)
    maps = pd.DataFrame(maps)   
    maps.columns = topks
    maps['distances'] =  distances
    maps['task'] = task
    return maps

def get_metadata(parameters):
    meta_data = json.load(open(parameters['data_prepo_meta'], 'r'))
    meta_data['ix_to_word'] = {str(word):int(i) for i,word in meta_data['ix_to_word'].items()}
    return meta_data

def prepare_embeddings(parameters,metadata):
    if os.path.exists(parameters['embedding_matrix_filename']):
        with h5py.File(parameters['embedding_matrix_filename']) as f:
            return np.array(f['embedding_matrix'])

    embeddings_index = {}
    with open(parameters['glove_path'], 'r') as glove_file:
        for line in glove_file:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    embedding_matrix = np.zeros((parameters['num_words'], parameters['embedding_dim']))
    word_index = metadata['ix_to_word']

    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
   
    with h5py.File(parameters['embedding_matrix_filename'], 'w') as f:
        f.create_dataset('embedding_matrix', data=embedding_matrix)

    return embedding_matrix

def configure_logger(LOGFILENAME):
    if os.path.exists(LOGFILENAME):
        os.remove(LOGFILENAME)
    logger = logging.getLogger("my_logger")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    fh = logging.FileHandler(LOGFILENAME,'w+')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

def get_my_model(parameters,metadata):
    #print("Creating Model...")
    num_words = len(metadata['ix_to_word'].keys())
    parameters['num_words'] = num_words    
    embedding_matrix = prepare_embeddings(parameters, metadata)        
    #print('num_words: ', parameters['num_words'])
    #print('num_classes:', parameters['num_classes'])
    #print('embedding_matrix dim: ',embedding_matrix.shape)    
    model = multimodal_net(embedding_matrix, parameters)
    return model

def get_rep(tokens,seq_length,metadata):    
    emb = []
    for token in tokens:
        if token in metadata['ix_to_word'].keys():
            emb.append(metadata['ix_to_word'].get(token))
    if len(emb) < seq_length:
        emb =  [0.0] * (seq_length-len(emb)) + emb       
    else:
        emb = emb[0:seq_length]
    return emb

def prep_input(img):
    img = np.expand_dims(img,axis=0).astype(float)
    img = preprocess_input(img) # take into account when different model than VGG
    return img

def merge_dicts(x, y):
    z = x.copy()  
    z.update(y)   
    return z

def sep_model(parameters,metadata,indl=0):     
    logger = logging.getLogger("my_logger")    
    model = get_my_model(parameters,metadata) 
    logger.info('Loading best model weights: %s',parameters['model_weights_filename'])
    model.load_weights(parameters['model_weights_filename'])
    
    #branch = Model(inputs=model.layers[br], outputs=model.layers[0].layers[br].output)   # keras 1.2.2
    branch = Model(inputs=model.get_layer(parameters['in']).input,output=model.get_layer(parameters['out']).output) #keras > 2.0
    
    # remove merge layer
    # model.layers.pop(0) # keras 1.2

    # joininig remaining layers
    l = model.layers[parameters['merged_ind']](branch.output)
    ''' # keras < 2.0
    outl = len(model.layers) - indl
    for i in range(1, outl):
        l = model.layers[i](l)
    '''
    for i in range(parameters['merged_ind']+1,len(model.layers)-indl):
        l = model.layers[i](l)
    
    stacked_model = Model(branch.input, l)
    stacked_model.summary()    
    return stacked_model


class eval_on_valCallback(Callback):
    def on_train_begin(self, logs=None):
        self.acc_val = []
        
    def on_train_end(self, logs=None):
        print(self.acc_val)
        
    def on_epoch_end(self,epoch,logs=None):
        x_1 = self.validation_data[0]
        x_2 = self.validation_data[1]
        y_test = self.validation_data[2]   
        
        print('Dims Validation data: %s %s %s',x_1.shape,x_2.shape,y_test.shape)
        # predicting outputs for val data
        y_pred = self.model.predict([x_1,x_2])
        
        # selecting the top value of predictions and
        y_test = np.argmax(y_test, axis=-1)
        y_pred = np.argmax(y_pred, axis=-1)       
        
        self.acc_val.append(acc(y_test,y_pred))
        print ('Acc: ',acc(y_test,y_pred))   
        
def evaluate_ret(parameters,metadata,features,GT_samples):  
    logger = logging.getLogger("my_logger")    
    logger.info('Evaluating on test - retrieval')
    
    start_time = time.time()

    features_v = features[0]
    features_t = features[1]
    
    logger.info('Creating image embeddings...')    
    #br - branche: 0 images, 1 text < keras 2.0
    parameters['in'] = 'dense_1_input'
    parameters['out'] = 'dense_1'          
    submodel = sep_model(parameters,metadata)  
    img_embeddings = {}            
    for k,im in features_v.items():
        # filtering data
        #if k in GT_samples.keys():
        if parameters['cnn'] == 'vgg':
            feat = im[0].reshape(1,len(im[0])) 
        elif parameters['cnn'] == 'resnet':
            feat = im.reshape(1,len(im))
        elif parameters['cnn'] == 'inc_resnet':
            feat = im.reshape(1,len(im[0]))
        else:
            feat = im.reshape(1,len(im[0]))
        pred = submodel.predict(feat)
        img_embeddings[k] = pred.squeeze()
    logger.info('Number of embeddings for images: %s', str(len(img_embeddings)))

    logger.info('Creating text embeddings...')
    parameters['in'] = 'embedding_1_input'
    parameters['out'] = 'dense_2'
    submodel = sep_model(parameters,metadata) 
    text_embeddings = {}
    for k,feat in features_t.items():
        pred = submodel.predict(np.asarray(feat).reshape(1,len(feat)))
        text_embeddings[k] = pred.squeeze()
    logger.info('Number of embeddings for texts: %s', str(len(text_embeddings)))    
                  
    distances = parameters['distances']
    #tops = [parameters['topk']]
    tops = [10,100,200,len(text_embeddings)]
    logger.info('Computing distances: %s',str(distances))    
    
    logger.info('Check Embedding dims: %s, %s',len(img_embeddings),len(text_embeddings))
    
    logger.info('CROSS-MODAL tops: %s',str(tops))
    maps1 = calc_distances(img_embeddings,text_embeddings,'img2txt',tops,distances,GT_samples)     
    maps2 = calc_distances(text_embeddings,img_embeddings,'txt2img',tops,distances,GT_samples)
    mapt_cross = pd.concat([maps1, maps2], axis=0)
    mapt_cross = mapt_cross[['task','distances']+tops]
    logger.info('MAPS txt2txt: %s',str(mapt_cross))
    
    tops = [10,100,200,len(text_embeddings)-1]
    logger.info('UNI-MODAL tops: %s',str(tops))
    maps3 = calc_distances(img_embeddings,img_embeddings,'img2img',tops,distances,GT_samples)
    maps4 = calc_distances(text_embeddings,text_embeddings,'txt2txt',tops,distances,GT_samples)
    mapt_uni = pd.concat([maps3, maps4], axis=0)
    mapt_uni = mapt_uni[['task','distances']+tops]
    logger.info('MAPS txt2txt: %s',str(mapt_uni))
    
    maps_tot = {}
    maps_tot['cross'] = mapt_cross
    maps_tot['uni'] = mapt_uni
    logger.info("Time for evaluation on retrieval %s seconds ---",str(time.time() - start_time))
    
    return maps_tot