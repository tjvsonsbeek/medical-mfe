from medical_metafeatures.feature_extraction import MetaFeatureExtraction
from medical_metafeatures.model_tuning import model_tune
from medical_metafeatures.encoder_decoder_networks import EncoderDecoderNetwork

from tqdm import tqdm
import numpy as np
import os
import sys
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def parse_args(args):

    """Parse command line parameters

    Args:
      args ([str]): command line parameters as list of strings

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-t",'--tasks', action='store', dest='tasks_list',
                    type=str, nargs='*', default=['Task04_Hippocampus'],
                    help="Examples: -i item1 item2, -i item3")
    parser.add_argument("--feature_extractors", action ='store',dest='feature_extractors', required=False, type=str, nargs='*', default = ['STAT','VGG16'], 
    help="feature extractors to use in a list. choose from 'STAT', 'VGG16', 'ResNet50' and  'MobileNetV1'")
    parser.add_argument("--load_labels", help='choose whether to load metalabels. will throw error if there are no metalabels. Currently only works for medical decathlon methods', default = False, type=bool)
    parser.add_argument("--meta_subset_size", default=20, help='nr of images on which metafeature is based',type=int)
    parser.add_argument("--meta_sample_size", default=1, help='nr of metafeatures ',type=int)
    parser.add_argument("--generate_model_weights", default=True, help='whether new model weights should be generated ',type=bool)
    parser.add_argument("--participants", required=False, default=['BCVuniandes', 'beomheep', 'CerebriuDIKU', 'EdwardMa12593', 'ildoo', 'iorism82', 'isarasua', 'Isensee', 'jiafucang', 'lesswire1', 'lupin', 'oldrich.kodym', 'ORippler', 'phil666', 'rzchen_xmu', 'ubilearn', 'whale', '17111010008', 'allan.kim01'], help='list of methods for which to extract the metalabels are loaded as well. input not relevant when --load_metalabels: True')
    parser.add_argument("--output_path", default= 'metafeature_extraction_result', type = str)
    parser.add_argument("--task_path", default= 'DecathlonData', type = str)

    parser.add_argument("--finetune_ntrain", default=800, help='number of training images in finetuning. Only applicable when generate_model_weights == True, default = 800 ',type=int)
    parser.add_argument("--finetune_nval", default=200, help='number of validation images in finetuning. Only applicable when generate_model_weights == True, default = 200',type=int)
    parser.add_argument("--finetune_nepoch", default=5, help='number of epochs in finetuning. Only applicable when generate_model_weights == True. default = 5 ',type=int)
    parser.add_argument("--finetune_batch", default=5, help='batch_size in finetuning. Only applicable when generate_model_weights == True. default = 5 ',type=int)
    return parser.parse_args(args)

def main(args):
    args = parse_args(args)
    tasks_list = args.tasks_list
    feature_extractors = args.feature_extractors
    load_labels = args.load_labels
    subset_size = args.meta_subset_size
    nr_of_subsets = args.meta_sample_size
    generate_model_weights = args.generate_model_weights
    output_path = args.output_path
    finetune_train_size = args.finetune_ntrain
    finetune_val_size = args.finetune_nval
    finetune_epochs = args.finetune_nepoch
    finetune_batchsize = args.finetune_batch
    
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    task_path = args.task_path
    if load_labels: participants = args.participants

    filters = {'VGG16': 512, 'MobileNetV1': 1024, 'ResNet50': 2048}

    custom_subset_size = {20: {'Task01_BrainTumour': 20,'Task02_Heart': 10,'Task03_Liver': 20,'Task04_Hippocampus': 20, 'Task05_Prostate': 16, 'Task06_Lung': 20, 'Task07_Pancreas': 20, 'Task08_HepaticVessel': 20, 'Task09_Spleen': 20, 'Task10_Colon': 20}}

    for task_id, task in enumerate(tasks_list):
        for fe in feature_extractors:

            ## some of the decathlon datasets have custom subset sizes to be able to compare robustness of the model consistently for smaller datasets as well.
            if subset_size in custom_subset_size.keys():
                if task in custom_subset_size[subset_size].keys():
                    subset_size = custom_subset_size[subset_size][task]

            if load_labels: labels = np.zeros((nr_of_subsets, subset_size, len(participants)))
            if fe == 'STAT':
                features = np.zeros((nr_of_subsets,subset_size, 33))
                model = None
                nr_of_filters = None
            else:
                nr_of_filters = filters[fe]
                features = np.zeros((nr_of_subsets,subset_size, 7,7,nr_of_filters))
                model = EncoderDecoderNetwork(fe,task_id, output_path)
                model.task = task
                model.build_encoder()
                model.build_decoder()

                if generate_model_weights:
                    model_tune(model, task, fe, task_path, output_path, finetune_train_size, finetune_val_size, finetune_epochs, finetune_batchsize)
                model.load_weights()
                model.update_encoder_weights()

            for subset in tqdm(range(nr_of_subsets)):

                m = MetaFeatureExtraction(task, subset_size, fe, model, nr_of_filters, task_path, output_path)
                if fe != 'STAT': m.load_model(model.feature_extractor)
                m.gather_random_addresses()
                m.gather_meta_features()

                if fe == 'STAT':
                    features[subset,:] = m.meta_features
                else:
                    features[subset,:,:,:,:] = m.meta_features
                if load_labels:
                    try:
                        m.gather_list_meta_labels()
                        labels[subset,:,:]   = m.meta_labels
                    except:
                        raise ValueError("No metalabels found at expected location")

            if load_labels:
                np.save(os.path.join(output_path, 'meta_regressor_labels_{}_{}.npy'.format(task, fe)), labels)
            np.save(os.path.join(output_path, 'meta_regressor_features_{}_{}.npy'.format(task, fe)), features)


def run():
    """Entry point for console_scripts
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
