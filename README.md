# DEEP MULTIMODAL EMBEDDINGS (DME)
This is an implementation of the model DME for the paper "Deep Multimodal learning for Cross-Modal Retrieval: onemodel for all tasks" in [Keras](https://keras.io/).

[Deep multimodal learning for cross-modal retrieval: One model for all tasks](https://www.sciencedirect.com/science/article/abs/pii/S0167865521000908) 

This repository is intended to provide a implementation of the paper for other researchers to build on. For more implementation details, please, check details on the paper.
  
## Download data
- Download each database from their corresponding sources
  - [MirFlickr](http://press.liacs.nl/mirflickr/mirdownload.html)
  - [Pascal](https://vision.cs.uiuc.edu/pascal-sentences/)
  - [Wikipedia](http://datasets.cvc.uab.es/rrc/wikipedia_data/)
  
- Download required files and located them in folder 'gen' of each database
  - [generated data](https://www.dropbox.com/sh/0zfp79u2h0o8l4m/AAAVkIgEgodwAclMRzgVIOcSa?dl=0) 

## Running the model
- Clone this repository as follows:

`git clone https://github.com/lvbeltranb/DME.git`

- Set the paths in most of the scripts:
  - src_path: path to the code
  - dataroot: path to the data
  
  ### Training
  - Modify params file in train_scripts according to the database
  - For each database, run the corresponding script in train_scripts
  - To use GPU, please uncomment lines in main files
  
  ### Evaluation
  Once the model is trained, run the corresponding train file and it will evaluate the model in the test database.
  
  ### Dependencies (more required)
  - Python > 3.5
  - Keras 2.3.1 (or see lines to change to run with a smaller version of keras).
  - sklearn 0.21.2
  - tensorflow 2.0.0
  - PIL 6.0.0
    
  ### Comments
  Output will be redirected to a logging file called summary.log in 'evals' path (check sample of log file in generated files link).
