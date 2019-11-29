Medical_metafeatures
===============

Toolkit for extraction of metafeatures from medical datasets. Four different methods for metafeature extraction can be used. 
## Description

Toolkit for extraction of metafeatures from medical datasets. 4 different methods for metafeature extraction can be used. (statistical, VGG16, ResNet50 and MobileNetV1).
Metafeatures are a compressed representation of a dataset which can be used for 

Images should have .nii.gz extension.
(if your images have .nii format, gzip: https://www.gzip.org/ can be used for fast conversion)

## Installation

If you are using pip:

    pip install medical_metafeatures

If you are using conda, you can install from the `conda-forge` channel:

    conda install -c conda-forge medical_metafeatures
## Dependencies

The main `medical_metafeatures` requirement is:
* Python (>= 3.6)

## Installation

The installation process is similar to other packages available on pip:

```python
pip install -U pymfe
```

It is possible to install the development version using:

```python
pip install -U git+https://github.com/ealcobaca/pymfe
```

or

```
git clone https://github.com/ealcobaca/pymfe.git
cd pymfe
python3 setup.py install
```
## Command-line usage

Use get_meta_features for the extraction of metafeatures. 

Example: 

    python -m medical_metafeatures.get_metafeatures --task 'Example_dataset' --feature_extractors 'STAT' 'VGG16', --meta_suset_size 15 --generate_weights False --output_path 'dest' --task_path 'datasets' 
   
___
Parameters for meta_get_features:
___
-t, --task

Name of dataset or datasets on which metafeatures will be extracted as string. Multiple inputs are possible
___
--feature_extractors
feature extractors to use for metafeature extraction. Expected as string.  choose from 'STAT', 'VGG16', 'ResNet50' and  'MobileNetV1'. Multiple inputs are possible. 
___
--load_labels
___
choose whether to load metalabels. will throw error if there are no metalabels. Currently only works for medical decathlon datasets. Metalabels are not public yet.
default = False

--output_path
____
--meta_subset_size
Nr of images on which metafeature is based
default = 20
____
--meta_sample_size
Number of metafeatures per dataset
default = 10
___
--generate_model_weights
Boolean which tells whether new model weights should be generated. Only used when deep learning based metafeature extraction is done. 
default = True
___
--output_path
Path where all output will be saved
___
--task_path

path in which to find the dataset folder. In this folder there should be a folder with the name of -t/--task. This folder should contain a ImagesTs folder with the images to extract the metafeature from in it. Images should have the .nii.gz extenstion

Metafeatures
===
Short explanation of 
___
Statistical
==
___
Deep learning based metafeatures
==
![](media/finetuning.png)
![](media/metalearningsystem.png)



Note
====

This project has been set up using PyScaffold 3.2.2. For details and usage
information on PyScaffold see https://pyscaffold.org/.