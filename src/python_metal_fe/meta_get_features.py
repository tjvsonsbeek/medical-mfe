from feature_extraction import MetaFeatureExtraction
from model_tuning import model_tune
from encoder_decoder_networks import EncoderDecoderNetwork

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
    parser.add_argument("tasks", help='list of tasks',type=list)
    parser.add_argument("--feature_extractors", required=False, default = ['STAT'], help="feature extractors to use in a list. choose between 'STAT', 'VGG16', 'ResNet50' and  'MobileNetV1'", type=list)
    parser.add_argument("--load_labels", help='choose whether to load metalabels. will throw error if there are no metalabels. Currently only works for medical decathlon methods', default = False, type=bool)
    parser.add_argument("--meta_subset_size", default=20, help='nr of subset in a metafeature',type=int)
    parser.add_argument("--meta_sample_size", default=1, help='nr of metafeatures ',type=int)
    parser.add_argument("--generate_model_weights", default=True, help='whether nwew model weights should be generated ',type=bool)
    parser.add_argument("--participants", required=False, default=['BCVuniandes', 'beomheep', 'CerebriuDIKU', 'EdwardMa12593', 'ildoo', 'iorism82', 'isarasua', 'Isensee', 'jiafucang', 'lesswire1', 'lupin', 'oldrich.kodym', 'ORippler', 'phil666', 'rzchen_xmu', 'ubilearn', 'whale', '17111010008', 'allan.kim01'], help='list of methods for which to extract the metalabels are loaded as well. input not relevant when --load_metalabels: True')
    parser.add_argument("--output_path", default= 'metafeature_extraction_result', type = str)
    parser.add_argument("--task_path", default= 'decathlonData', type = str)
    return parser.parse_args(args)

def main(args):
    args = parse_args(args)
    tasks_list = args.tasks
    feature_extractors = args.feature_extractors
    load_labels = args.load_labels
    subset_size = args.meta_subset_size
    nr_of_subsets = args.meta_sample_size
    generate_model_weights = args.generate_model_weights
    output_path = args.output_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    task_path = args.task_path
    if load_labels: participants = args.participants

    filters = {'VGG16': 512, 'MobileNetV1': 1024, 'ResNet50': 2048}

    custom_subset_size = {5: {'Task01_BrainTumour': 5,'Task02_Heart': 5,'Task03_Liver': 5,'Task04_Hippocampus': 5, 'Task05_Prostate': 5, 'Task06_Lung': 5, 'Task07_Pancreas': 5, 'Task08_HepaticVessel': 5, 'Task09_Spleen': 5, 'Task10_Colon': 5}, 10: {'Task01_BrainTumour': 10,'Task02_Heart': 6,'Task03_Liver': 10,'Task04_Hippocampus': 10, 'Task05_Prostate': 10, 'Task06_Lung': 10, 'Task07_Pancreas': 10, 'Task08_HepaticVessel': 10, 'Task09_Spleen': 10, 'Task10_Colon': 10}, 20: {'Task01_BrainTumour': 20,'Task02_Heart': 7,'Task03_Liver': 20,'Task04_Hippocampus': 20, 'Task05_Prostate': 11, 'Task06_Lung': 20, 'Task07_Pancreas': 20, 'Task08_HepaticVessel': 20, 'Task09_Spleen': 16, 'Task10_Colon': 20}}

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
                model = EncoderDecoderNetwork(fe,task_id, task_path, output_path)
                model.task = task
                model.build_encoder()
                model.build_decoder()

                if generate_model_weights:
                    model_tune(model, task, fe, task_path, output_path)
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
