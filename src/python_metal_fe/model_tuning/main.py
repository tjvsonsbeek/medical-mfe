from keras.optimizers import Adam
from python_metal_fe.utils import dice_coef_loss, auc, mean_iou
from keras.callbacks import ModelCheckpoint, EarlyStopping
import nibabel as nib
from scipy import misc
import cv2
import os
from tqdm import tqdm
import numpy as np

def makejpg(task, result_folder, size, task_path, output_path, image_dimensions = (224, 224)):
    count = 0
    addresses = os.listdir(os.path.join(task_path,task,'imagesTs'))[:20]
    images_per_address = np.ceil(size/len(addresses)).astype(np.uint8)
    for address in tqdm(addresses):
        img =  (nib.load(os.path.join(task_path,task,'imagesTs', address))).get_data()
        if len(img.shape) == 4:
            channel = np.random.choice(range(img.shape[3]))
            img = img[:,:,:,channel]
        for i in range(images_per_address):
            z = np.random.randint(int(img.shape[2]*0.25), int(img.shape[2]*0.75))
            misc.imsave(os.path.join(output_path, '{}_{}/images/{}.png'.format(result_folder,task,count)),  cv2.resize(img[:,:,z].astype('float32'),(224,224)))
            count+=1

def load_data(task, train_or_valid, path):
    addresses_list = []
    addresses = os.listdir(os.path.join(path, '{}_{}/images/'.format(train_or_valid, task)))
    for address in addresses:
        if address !='labels':
            addresses_list.append(os.path.join(path, '{}_{}/images/{}'.format(train_or_valid, task, address)))
    return addresses_list

def model_tune(enc_dec_model, task, feature_extractor, task_path,  out_path, train_size = 10, val_size = 4, epochs = 4, minibatch_size = 5, image_dimensions = (224,224)):
    ## create folder for train and validation images
    output_path = os.path.join(out_path, 'model_tune_data')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    ## check if train and validation images are already imageDimensions
    for i in ['train', 'valid']:
        if not os.path.isdir(os.path.join(output_path,"{}_{}".format(i ,task))):
            os.mkdir(os.path.join(output_path,"{}_{}".format(i ,task)))
        if not os.path.isdir(os.path.join(output_path,"{}_{}".format(i ,task),"images")):
            os.mkdir(os.path.join(output_path,"{}_{}".format(i ,task),"images"))
        if not os.path.isdir(os.path.join(output_path,"{}_{}".format(i ,task),"labels")):
            os.mkdir(os.path.join(output_path,"{}_{}".format(i ,task),"labels"))
        if i=='train':
            makejpg(task, i, train_size, task_path, output_path)
        elif i=='valid':
            makejpg(task, i, val_size, task_path, output_path)
    ## load the data addresses
    train_data = load_data(task, 'train', output_path)
    valid_data = load_data(task, 'valid', output_path)

    ## load encoderdecoder network and set parameters
    enc_dec_model.epochs = epochs
    enc_dec_model.minibatch_size = 5
    enc_dec_model.model.compile(optimizer = Adam(lr = 1e-5), loss = dice_coef_loss, metrics = ['accuracy', auc, mean_iou])

    ## train the model
    enc_dec_model.train(train_data, valid_data, image_dimensions, verbosity = 1)

    ## save the model
    enc_dec_model.save_model()
