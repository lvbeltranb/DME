import sys
import os
src_path = 'path_to_src'
dataroot = 'path_to_data'
# add src_path if it's located in another path
#sys.path.append(os.path.join(src_path,'multimodal_retrieval/'))
from utilities import *
from params import create_params,run_it

def prep_data(data_v,data_t,ind,parameters,metadata,GT):
    ind_train_X = [s[1] for s in ind['train'][0]]
    ind_val_X = [s[1] for s in ind['val'][0]]

    train_y_cl = {s[1]:c for s,c in zip(ind['train'][0],ind['train'][1])}
    train_y_cl_2 = {s[0]:c for s,c in zip(ind['train'][0],ind['train'][1])}
    train_y_cl = merge_dicts(train_y_cl, train_y_cl_2)
    val_y_cl = {s[1]:c for s,c in zip(ind['val'][0],ind['val'][1])}    
    
    features_v = {}
    for im in data_v:
        if os.path.exists(parameters['img_root']+im):
            img = cv2.imread(parameters['img_root']+im)
            img = cv2.resize(img,dsize=(parameters['MODEL_INPUT_SIZE'],parameters['MODEL_INPUT_SIZE']), 
                             interpolation=cv2.INTER_CUBIC)
            features_v[im.replace('.jpg','')] = img 
            
    # filtering texts if corresponde image exists
    filtered = [v for k,v in GT.items() if k in features_v.keys()]
    
    with open(data_t,'rb') as f:
        text = pickle.load(f)        
    features_t = {}
    for k,v in text.items():
        if k in filtered:
            features_t[k] = get_rep(v,parameters['seq_length'],metadata)
        
    train_X_v = []    
    train_X_t = []
    train_y = []
    val_X_v = []
    val_X_t = []
    val_y = []
    
    for s in features_v.keys():
        # if s is a train sample and also, if the image exists
        if s in ind_train_X:
            train_X_v.append(prep_input(features_v.get(s)).squeeze()) #the rep comes with at least the required size
            train_X_t.append(features_t.get(GT.get(s))[0:parameters['seq_length']])
            train_y.append(to_categorical(train_y_cl.get(s),parameters['num_classes']))
        else:
            val_X_v.append(prep_input(features_v.get(s)).squeeze())
            val_X_t.append(features_t.get(GT.get(s))[0:parameters['seq_length']])
            val_y.append(to_categorical(val_y_cl.get(s),parameters['num_classes']))
            
    print('Training before computing pretrained: %s, %s:', str(np.asarray(train_X_v).shape),str(np.asarray(train_y).shape))
    print('Validation before computing pretrained: %s, %s:', str(np.asarray(val_X_v).shape),str(np.asarray(val_y).shape))      
    
    if parameters['use_pretrained']:
        print('Computing embeddings from pretrained model: %s', parameters['cnn'])
        if not os.path.exists(parameters['embeddings_pretrained_test']):
            print('Computing embeddings from pretrained model for test images: %s', parameters['cnn'])
            type_data = 'test'
            im_txt_pair_wd = open(parameters['datapath'] +'testset_txt_img_cat.list', 'r').readlines()
            # 0-->text, 1-->images, 2-->class
            test_img_files = [i.split('\t')[1] + '.jpg' for i in im_txt_pair_wd]
            test_img = {}
            for im in test_img_files:
                img = cv2.imread(parameters['img_root']+im)
                img = cv2.resize(img,dsize=(parameters['MODEL_INPUT_SIZE'],parameters['MODEL_INPUT_SIZE']), 
                                     interpolation=cv2.INTER_CUBIC)
                test_img[im.replace('.jpg','')] = prep_input(img)        
        
            train_X_v,val_X_v = compute_embeddings_pretrained(parameters,np.asarray(train_X_v), np.asarray(val_X_v),test_img)   
        else:
            print('Computing embeddings from pretrained model only for train and val: %s', parameters['cnn'])
            train_X_v,val_X_v = compute_embeddings_pretrained(parameters,np.asarray(train_X_v), np.asarray(val_X_v))
    
    train_X_t = np.asarray(train_X_t)
    val_X_t = np.asarray(val_X_t)
    train_y = np.asarray(train_y)
    val_y = np.asarray(val_y)    
    
    ###----------Joining training and validation data
    train_X_v = np.vstack((train_X_v,val_X_v))
    train_X_t = np.vstack((train_X_t,val_X_t))    
    train_y = np.vstack((train_y,val_y))
    ###----------------------------------------------
 
    train_X = [train_X_v, train_X_t]
    val_X = [val_X_v, val_X_t]

    return train_X,train_y,val_X,val_y


def train(): 
    datapath = os.path.join(dataroot,'datasets/wikipedia/')
    pathgen = os.path.join(dataroot,'datasets/wikipedia/gen/')       
    num_cl = 10
    parameters = create_params(dataroot,datapath,pathgen,num_cl)
    
    # logger to file...
    logfilename = os.path.join(parameters['eval'],'summary.log')
    print('LOG FILENAME: ', logfilename)
    logger = configure_logger(logfilename)
    logger.info('TRAINING - WIKIPEDIA RETRIEVAL DATASET ')
    
    # parameters for model
    parameters['acc'] = categorical_accuracy   
    parameters['img_root'] = datapath + '/images_wd_256/'
    parameters['embeddings_pretrained_test'] = pathgen + '/resnet_all.pkl'
    logger.info('General parameters: %s', str(parameters))  
##############################################################################  
    logger.info('Reading and preprocessing data...')    
    logger.info('Reading indexes file for partitions in train and val: %s')
    with open(os.path.join(datapath,'indexes.pkl'),'rb') as f:
        ind = pickle.load(f)
        
    # 0-->text, 1-->images, 2-->class   
    im_txt_pair_wd = open(datapath +'/trainset_txt_img_cat.list', 'r').readlines()
    GT_img2txt = {}     
    for i in im_txt_pair_wd:
        GT_img2txt[i.split('\t')[1]] = i.split('\t')[0] #Corresponding image
     
    data_v = [i.split('\t')[1] + '.jpg' for i in im_txt_pair_wd]           
    # texts are vectors of words beloging to each sample and metadata file
    data_t = pathgen + 'vectors_l1.pkl'
    
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
    
    train_X,train_y,val_X,val_y = prep_data(data_v,data_t,ind,parameters,metadata,GT_img2txt)
    logger.info('Dimensions for data after augmentation and computing pretrained model:')
    logger.info('Training: %s, %s, %s', str(train_X[0].shape),str(train_X[1].shape),str(train_y.shape))
    logger.info('Validation: %s, %s, %s', str(val_X[0].shape),str(val_X[1].shape),str(val_y.shape))
    
    run_it(parameters,logger,metadata,train_X, train_y)       
                                           
        
if __name__ == '__main__':
    train()