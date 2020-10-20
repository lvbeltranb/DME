import os
import sys
src_path = 'path_to_src'
dataroot = 'path_to_data'
# add src_path if it's located in another path
#sys.path.append(os.path.join(src_path,'multimodal_retrieval/'))
from utilities import *
# Change evaluation method according to database: evaluate_ret_wiki or evaluate_ret for Pascal/Mirflick
# Change topk values according to database

def evaluate_ret_wiki(parameters,metadata):    
    logger = logging.getLogger("my_logger")    
    logger.info('Evaluating on test - retrieval')
    
    start_time = time.time()
    
    logger.info('Reading the images...')    
    features_v = []  
    with open(parameters['embeddings_pretrained_test'], 'rb') as handle:
        features_v = pickle.load(handle)

    logger.info('Reading the texts...') 
    data_test = parameters['datagen'] + 'test_l1.pkl'
    with open(data_test,'rb') as f:
        data = pickle.load(f)              
    features_t = {}
    for k,v in data.items():
        features_t[k] = get_rep(v,parameters['seq_length'],metadata)

    im_txt_pair_wd = open(parameters['datapath'] +'testset_txt_img_cat.list', 'r').readlines() # Image-text pairs
    GT_samples = {} # In general
    for i in im_txt_pair_wd:                
        GT_samples[i.split('\t')[1]] = i.split('\t')[2].replace('\n','') # (Corresponding text, class)
        GT_samples[i.split('\t')[0]] = i.split('\t')[2].replace('\n','') # (Corresponding image, class)    
    
    logger.info('Creating image embeddings...')    
    #br - branche: 0 images, 1 text < keras 2.0
    parameters['in'] = 'dense_1_input'
    parameters['out'] = 'dense_1'          
    submodel = sep_model(parameters,metadata)  
    img_embeddings = {}            
    for k,im in features_v.items():
        # filtering data
        if k in GT_samples.keys():
            if parameters['cnn'] == 'vgg':
                feat = im[0].reshape(1,len(im[0])) 
            elif parameters['cnn'] == 'resnet':
                feat = im.reshape(1,len(im[0]))
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
    tops = parameters['topk']
    logger.info('Computing distances: %s',str(distances))    
    
    logger.info('Check Embedding dims: %s, %s',len(img_embeddings),len(text_embeddings))
    
    logger.info('CROSS-MODAL tops: %s',str(tops))
    maps1 = calc_distances(img_embeddings,text_embeddings,'img2txt',tops,distances,GT_samples)     
    maps2 = calc_distances(text_embeddings,img_embeddings,'txt2img',tops,distances,GT_samples)
    mapt_cross = pd.concat([maps1, maps2], axis=0)
    mapt_cross = mapt_cross[['task','distances']+tops]
    logger.info('MAPS txt2txt: %s',str(mapt_cross))

    tops = tops[:-1] + [tops[-1]-1]
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

def run_it(parameters,logger,metadata,train_X, train_y,val_y=None,val_r=None,GT_r=None):    
    
    for opt in parameters['optimizers']:
        parameters['optimizer'] = opt
        for bs in parameters['batch_sizes']:
            for dout in parameters['dropout']:
                parameters['dropout_rate'] = dout
                logger.info('optimizer: %s, batch_size: %s, dropout: %s',opt,bs,dout)
                start_time = time.time()

                modelname = opt+'_'+str(bs)+'_'+str(dout)+'.npy'                                      
                logger.info('modelname: %s',modelname)
                
                parameters['ckpt_model_weights_filename'] = os.path.join(parameters['eval'],modelname.replace('.npy','')+'_weights.h5')
                parameters['model_weights_filename'] = os.path.join(parameters['eval'],modelname.replace('.npy','')+'_weights.h5') 
                checkpointer = ModelCheckpoint(filepath=parameters['ckpt_model_weights_filename'],verbose=1,monitor='val_categorical_accuracy', save_best_only=True,mode='auto')
                eval_val = eval_on_valCallback()                    

                logger.info('Current parameter configuration: %s',str(parameters))
                
                if os.path.exists(os.path.join(parameters['eval'],modelname)):
                    logger.info('Model already trained...')
                    pass
                else:
                    logger.info('Training model...')
                    model = get_my_model(parameters,metadata)                        
                    #history = model.fit(train_X, train_y, epochs=parameters['epochs'], batch_size=bs, validation_data=(val_X,val_y), callbacks=[checkpointer,eval_val], shuffle="batch")
                    history = model.fit(train_X, train_y, epochs=parameters['epochs'], batch_size=bs)
                    
                    model.save_weights(parameters['model_weights_filename'],overwrite=True)
                    np.save(os.path.join(parameters['eval'],modelname),history.history)                     
                    logger.info('Finished training...')  
                    logger.info("Time for evaluation on training: %s seconds ---",str(time.time() - start_time))  
                    
                if parameters['eval_model_ret']:
                    logger.info('Evaluating best model over retrieval: %s')                      
                    results_ret = open(os.path.join(parameters['eval'],'eval_retrieval.txt'),"a") 

                    results_ret.write(modelname.replace('.npy','')+'\n') 
                    #results_ret.write(str(evaluate_ret_wiki(parameters,metadata))) # wiki
                    results_ret.write(str(evaluate_ret(parameters,metadata,val_r,GT_r))) # pascal/mirflick
                    results_ret.write('\n-------------------------------------------\n')
                    results_ret.close() 

def create_params(dataroot,datapath,pathgen,nclasses):
    
    # path to save weights and results of evaluation
    eval_p = pathgen+'/evals/' 
    if not os.path.exists(eval_p):
        os.makedirs(eval_p)
        
    parameters = {} 
    parameters['eval'] = eval_p
    parameters['epochs'] = 200
    # path where data files are stored
    parameters['datapath'] = datapath
    # path where preprocessed and generated data is found
    parameters['datagen'] = pathgen
    
    parameters['batch_sizes'] = [256]
    parameters['dropout'] =  [0.3]
    parameters['optimizers'] = ['rmsprop']

    # parameters for text
    parameters['seq_length'] = 100  
    # parameters for text
    parameters['texts'] = os.path.join(pathgen,'texts.pkl')  
    parameters['data_prepo_meta'] = os.path.join(pathgen,'data_prepro.json')

    #embedding matrix name
    emb_name = 'glove'
    parameters['embedding_dim']= 300
    parameters['glove_path'] = os.path.join(dataroot,'datasets/text_models/glove.6B.300d.txt')
    parameters['embedding_matrix_filename'] = os.path.join(pathgen,'embeddings_%s'%parameters['embedding_dim']+emb_name+'.h5')

    # parameters for images
    parameters['img_root'] = os.path.join(datapath,'images/')   
    parameters['MODEL_INPUT_SIZE'] = 224 # 299-inc_resnet, Xception; 224 vgg, resnet, densenet
    parameters['use_pretrained'] = True
    parameters['cnn'] = 'resnet' # vgg, densenet, resnet, inc_resnet, xception
    parameters['dim_input_cnn'] = 2048 #2048 ResNet, 4096 VGG, 1024 Densenet, 1536 inc_resnet, 2048 Xception
    parameters['pool'] = 'avg'
    parameters['embeddings_pretrained'] = pathgen + '/resnet_all.pkl'

    # parameters for model
    act = 'relu'
    dim_h = 256
    parameters['loss'] = 'binary_crossentropy' 
    parameters['num_classes'] = nclasses
    parameters['layers'] = [(dim_h,act),(dim_h,act),(parameters['num_classes'],'softmax')] 
    parameters['layers_im'] = [(dim_h,act)]
    parameters['layers_tx'] = [(dim_h,act)]

    # change if number of layers before merge changes...
    parameters['merged_ind'] = 10 
    
    # distance to use to find closest embeddings
    parameters['distances'] = ['corr']
    
    # how many samples use to evaluate retrieval performance
    #parameters['topk'] = [8,50,500,693] # wiki
    parameters['topk'] = [10,100,200] # Pascal/Mirflick
    
    # percentage of gpu to use
    parameters['gpu_p'] = 0.5
    
    # evaluate retrieval performance
    parameters['eval_model_ret'] = True

    return parameters