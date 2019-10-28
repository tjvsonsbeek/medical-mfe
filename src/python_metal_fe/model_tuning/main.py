from keras.optimizers import Adam
from utils import dice_coef_loss, auc, mean_iou
from keras.callbacks import ModelCheckpoint, EarlyStopping
import nibabel as nib
from scipy import misc
import cv2
import os
from tqdm import tqdm
import numpy as np

def makejpg(task, result_folder, size, image_dimensions = (224, 224),path = "/home/tjvsonsbeek/featureExtractorUnet/decathlonDataProcessed/"):
    count = 0
    addresses = os.listdir(os.path.join('/home/tjvsonsbeek/decathlonData',task,'imagesTs'))
    images_per_address = np.ceil(size/len(addresses)).astype(np.uint8)
    for address in tqdm(addresses):
        img =  (nib.load(os.path.join('/home/tjvsonsbeek/decathlonData/'+task+'/imagesTs', address))).get_data()
        if len(img.shape) == 4:
            channel = np.random.choice(range(img.shape[3]))
            img = img[:,:,:,channel]
        for i in range(images_per_address):
            z = np.random.randint(int(img.shape[2]*0.25), int(img.shape[2]*0.75))
            misc.imsave('/home/tjvsonsbeek/featureExtractorUnet/decathlonDataProcessed/{}_{}/images/{}.png'.format(result_folder,task,count),  cv2.resize(img[:,:,z].astype('float32'),(224,224)))
            count+=1

def load_data(task, train_or_valid):
    addresses_list = []
    addresses = os.listdir('/home/tjvsonsbeek/featureExtractorUnet/decathlonDataProcessed/{}_{}/images/'.format(train_or_valid, task))
    for address in addresses:
        if address !='labels':
            addresses_list.append('/home/tjvsonsbeek/featureExtractorUnet/decathlonDataProcessed/{}_{}/images/{}'.format(train_or_valid, task, address))
    return addresses_list

def model_tune(enc_dec_model, task, feature_extractor, train_size = 1000, val_size = 400, epochs = 4, minibatch_size = 5, image_dimensions = (224,224)):
    path = '/home/tjvsonsbeek/featureExtractorUnet/decathlonDataProcessed'
    ## check if train and validation images are already imageDimensions
    for i in ['train', 'valid']:
        if not os.path.isdir(os.path.join(path,"{}_{}".format(i ,task))):
            os.mkdir(os.path.join(path,"{}_{}".format(i ,task)))
        if not os.path.isdir(os.path.join(path,"{}_{}".format(i ,task),"images")):
            os.mkdir(os.path.join(path,"{}_{}".format(i ,task),"images"))
        if not os.path.isdir(os.path.join(path,"{}_{}".format(i ,task),"labels")):
            os.mkdir(os.path.join(path,"{}_{}".format(i ,task),"labels"))
        if i=='train':
            makejpg(task, i, train_size)
        elif i=='valid':
            makejpg(task, i, val_size)
    ## load the data
    train_data = load_data(task, 'train')
    valid_data = load_data(task, 'valid')

    ## load encoderdecoder network and set parameters
    enc_dec_model.epochs = epochs
    enc_dec_model.minibatch_size = 5
    enc_dec_model.model.compile(optimizer = Adam(lr = 1e-5), loss = dice_coef_loss, metrics = ['accuracy', auc, mean_iou])

    ## train the model
    enc_dec_model.train(train_data, valid_data, image_dimensions, verbosity = 1)

    ## save the model
    enc_dec_model.save_model()
