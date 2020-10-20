import sys
import os
src_path = 'path_to_src'
dataroot = 'path_to_data'
# add src_path if it's located in another path
#sys.path.append(os.path.join(src_path,'multimodal_retrieval/'))
from utilities import *
from params import create_params, run_it

def prep_data(ind,parameters,metadata,GT):    
    
    ind_train_X = ind['train']
    ind_val_X = ind['val']    
   
    # correspondance between id and img/txt code
    with open(os.path.join(parameters['datagen'],'idS_id_im.pkl'), 'rb') as f:
        ids_i = pickle.load(f)
    # correspondance between id and img/txt code
    with open(os.path.join(parameters['datagen'],'idS_id_tx.pkl'), 'rb') as f:
        ids_t = pickle.load(f)
    
    #features_v = {}
    with open(parameters['embeddings_pretrained'],'rb') as f:
        features_v = pickle.load(f)
    #for k,v in features_embs.items():
    #    features_v[ids_i.get(k)] = v 

    features_t = {}
    with open(parameters['texts'],'rb') as f:
        texts = pickle.load(f)   
        
    for k,v in texts.items():
        features_t[k] = get_rep(v,parameters['seq_length'],metadata)
    #for k,v in texts.items():
    #    features_t[ids_i.get(k)] = get_rep(v,parameters['seq_length'],metadata)            
    
    # vectors with multilabel representation for target, [1,0,1,0,...]
    with open(parameters['target'], 'rb') as f:
        target = pickle.load(f) 
    
    train_X_v = []    
    train_X_t = []
    train_y = []
    val_X_v = []
    val_X_t = []
    val_y = []
    
    # for retrieval
    val_xv_r = {}
    val_xt_r = {}
    
    # getting indexes for each partition
    ind_train_X = ind['train']
    ind_val_X = ind['val']
    
    for s in features_v.keys():
        if s in list(texts.keys()) and s in list(target.keys()):       
            # if s is a train sample and also, if the image exists
            if s in ind_train_X: # not all samples has representation            
                train_X_v.append(prep_input(features_v.get(s)).squeeze()) #the rep comes with at least the required size
                train_X_t.append(features_t.get(s)[0:parameters['seq_length']]) #the rep comes with at least the required size
                #train_y.append(to_categorical(train_y_cl.get(s),parameters['num_classes']))
                train_y.append(target.get(s))
            else:                
                val_X_v.append(prep_input(features_v.get(s)).squeeze())
                val_X_t.append(features_t.get(s)[0:parameters['seq_length']])
                #val_y.append(to_categorical(val_y_cl.get(s),parameters['num_classes']))  
                val_y.append(target.get(s))

                val_xv_r[ids_i.get(s)] =  val_X_v[-1]
                val_xt_r[ids_t.get(s)] =  val_X_t[-1]

    train_X_v = np.asarray(train_X_v)
    val_X_v = np.asarray(val_X_v)
    train_X_t = np.asarray(train_X_t)
    val_X_t = np.asarray(val_X_t)
    train_y = np.asarray(train_y)
    val_y = np.asarray(val_y)    
    
    train_X = [train_X_v, train_X_t]
    val_X = [val_X_v, val_X_t]
    
    val_r = [val_xv_r,val_xt_r]

    return train_X,train_y,val_X,val_y, val_r


def my_train():

    datapath = os.path.join(dataroot,'datasets/mirflick25/')
    pathgen = os.path.join(dataroot,'datasets/mirflick25/gen/')   
    num_cl = 24
    parameters = create_params(dataroot,datapath,pathgen,num_cl)
    
    # logger to file...
    logfilename = os.path.join(parameters['eval'],'summary.log')
    print('LOG FILENAME: ', logfilename)
    logger = configure_logger(logfilename)
    logger.info('TRAINING EXPLORATION - Mirflick25k DATASET ') 
    # specifics
    parameters['target'] =  os.path.join(pathgen,'target_onehot.pkl')       
    # parameters for model
    parameters['acc'] = categorical_accuracy   
    logger.info('General parameters: %s', str(parameters))
#############################################################################   
    logger.info('Reading and preprocessing data...')    
    logger.info('Reading indexes file for partitions in train and val: %s')
    with open(os.path.join(parameters['datagen'],'indexes6.pkl'),'rb') as f:
        ind = pickle.load(f) #ind['train/val']

    # file with correspondance: idx_sample: class    
    with open(os.path.join(parameters['datagen'],'idx_cat.pkl'), 'rb') as f:
        GT = pickle.load(f)    
   
    # For retrieval
    with open(os.path.join(parameters['datagen'],'txt_img_cat.pkl'), 'rb') as f:
        GT_r = pickle.load(f)
    
    metadata = get_metadata(parameters)
    logger.info('Num_words according to metadata file: %s',len(metadata['ix_to_word'].keys()))
    
    logger.info('Setting config for GPU...')
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=parameters['gpu_p'])     
    #sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    #sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    
    #config = ConfigProto()
    #config.gpu_options.per_process_gpu_memory_fraction = parameters['gpu_p']
    #config.gpu_options.allow_growth = True
    #session = InteractiveSession(config=config)    
    
    train_X,train_y,val_X,val_y,val_r = prep_data(ind,parameters,metadata,GT)
    logger.info('Dimensions for data:')
    logger.info('Training: %s, %s, %s', str(train_X[0].shape),str(train_X[1].shape),str(train_y.shape))
    logger.info('Validation: %s, %s, %s', str(val_X[0].shape),str(val_X[1].shape),str(val_y.shape))
    logger.info('Retrieval: %s, %s, %s', str(len(val_r[0])),str(len(val_r[1])))

    run_it(parameters,logger,metadata,train_X, train_y,val_y,val_r,GT_r)   
if __name__ == '__main__':
    my_train()
