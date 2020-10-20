import keras
import numpy as np
import pandas as pd
import random
import cv2
from PIL import Image
from skimage.io import imread
from skimage.color import grey2rgb
import pickle
import os 
import sys
from keras.utils import Sequence
#my libraries
src_path = 'path'
# add src_path if it's located in another path
#sys.path.append(os.path.join(src_path,'multimodal_retrieval/'))
from utilities import *

'''
IMAGES
'''
def fix_img(img,params):
    ds = params['img_size']
    sh = img.shape[0:2]
    wc,wp = check_dim(sh[0],ds)
    hc,hp = check_dim(sh[1],ds)
    #print wc,wp,hc,hp
    img = img[wc[0]:wc[1],hc[0]:hc[1],:]
    #print img.shape
    
    #cv2.BORDER_REPLICATE,cv2.BORDER_REFLECT,cv2.BORDER_REFLECT_101,cv2.BORDER_WRAP
    color = [0, 0, 0]
    # dst, top, bottom, left, right, borderType, value );
    img = cv2.copyMakeBorder(img,0,wp,0,hp,cv2.BORDER_REFLECT_101)#,value=color)
    
    return img


def get_i(image_path,params,aug=False):    
    img = cv2.imread(os.path.join(params['images_root'],image_path),cv2.IMREAD_COLOR)
    #img = imread(os.path.join(params['images_root'],image_path))
    
    '''
    if len(img.shape)<3: # converting image to rgb
        img = grey2rgb(img)
    '''
    img = fix_img(img,params)
    # change this to use a cropping image, to not loss 
    #img = cv2.resize(img,dsize=(params['img_size'],params['img_size']), 
    #                         interpolation=cv2.INTER_CUBIC)
    
    # apply transformations as a strategy of data augmentation
    
    if aug: # apply this at the beginning, only to training images
        transf, perm = get_trans_perm()
        set_trs = np.array(perm)[np.sort(random.sample(range(len(perm)), 1))]
        #print 'transf: ',set_trs
        for tr in set_trs:    
            for t in tr:
                img = transf[t](img)    
    return img

def get_i_emb(image_path,params):
    with open(os.path.join(params['pretrained_imgs'],image_path.replace('.jpg','.pkl')),'rb') as f:
        emb = pickle.load(f)
    return emb.squeeze()

'''
TEXT
'''
def get_rep(tokens,seq_length,metadata):    
    emb = []
    for token in tokens:
        if token in metadata['ix_to_word'].keys():
            emb.append(metadata['ix_to_word'].get(token))
    if len(emb) < seq_length:
        emb =  [0.0] * (seq_length-len(emb)) + emb       
    else:
        emb = emb[0:seq_length]
    return emb[0:seq_length]

def get_t(tokens,params):
    metadata = get_metadata(params)
    return get_rep(tokens,params['seq_length'],metadata)

'''
ANSWERS -Ys
'''
def get_ngram(word,n):
    return [word[i:i+n] for i in range(len(word)-n+1)]

def get_tngrams(word,ngram_levels):
    t_ngrams = []
    for n in ngram_levels:
        t_ngrams+= get_ngram(word,n)
    return t_ngrams

def get_y(answer,levels,total_ngrams): 
    ngrams = get_tngrams(answer,levels)
    phoc = np.zeros(len(total_ngrams))
    
    for n in ngrams:
        if n in total_ngrams: 
            #print n
            phoc[total_ngrams.index(n)] = 1 
    return phoc 

class DataGenerator(Sequence):
    def __init__(self,params,mode='train',aug=False,shuffle=False):
        'Initialization'
        self.params = params
        self.mode = mode
        self.aug = aug
        self.shuffle = shuffle
            
        self.data_i = pd.read_pickle(self.params['datapath_i'])     
       
        self.list_IDs = self.data_i.question_id.tolist()
        
        self.images_emb = self.params['pretrained_imgs']

        if self.mode in ['val','test']:
            #self.batch_size = 2
            self.batch_size = len(self.data_i.question)
        else:
            self.batch_size = self.params['bs']
            
        self.total_ngrams = np.load(params['tot_ngrams']).tolist()   
        self.on_epoch_end()
        
        print('Number of samples in data: ', len(self.list_IDs))

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch        
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        #print 'LIST IDS:',len(list_IDs_temp)
        
        # Generate data
        if self.mode=='train':
            X,y = self.__data_generation(list_IDs_temp)  
            return X,y
        
        elif self.mode=='val':
            X,y,batch_ids,words = self.__data_generation(list_IDs_temp)   
            return X,y,batch_ids,words
        
        else:# test samples
            X, batch_ids = self.__data_generation(list_IDs_temp)
            return X, batch_ids

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_ids):
        'Generates data containing batch_size samples'
        # Initialization
        images = []
        texts = []
        ys = []
        words = []

        # Generate data
        for input_s in batch_ids:               
            images += [get_i_emb(self.data_i[self.data_i['question_id'] == input_s].file_path.values[0],self.params)]
            
            if self.params['wmi']:
                texts += [get_t(self.data_i[self.data_i['question_id'] == input_s].question_tokens.values[0],self.params)]
            else:
                # get vector from vectors data if bert or elmo embeddings
                #texts += [self.data_t.get(input_s).squeeze()]
                pass
            
            if self.mode in ['train','val']:
                answers = self.data_i[self.data_i['question_id'] == input_s].answers.values[0]
                #print answers
                ys += [get_y(''.join(answers).replace(' ',''),self.params['ngram_levels'],self.total_ngrams)]
                words += [answers]            

        # Return a tuple of (input,output) to feed the network
        images = np.asarray(images)
        texts = np.asarray(texts)
        
        if self.mode in ['train','val']:
            ys = np.asarray(ys)             
            
        if self.mode=='train':
            return [images,texts],ys
        elif self.mode=='val':
            return [images,texts],ys,batch_ids,words
        else:# test samples
            return [images,texts],batch_ids
        