__author__ = "tjvsonsbeek"
__copyright__ = "tjvsonsbeek"
__license__ = "mit"


import argparse
from argparse import ArgumentParser
import genericpath
from mock import patch
import ntpath
import numpy as np
import numpy.lib.npyio
from numpy.lib.npyio import BagObj
import os
from python_metal_fe.encoder_decoder_networks import EncoderDecoderNetwork
import python_metal_fe.encoder_decoder_networks.main
from python_metal_fe.encoder_decoder_networks.main import Conv2D
from python_metal_fe.feature_extraction import MetaFeatureExtraction
import python_metal_fe.feature_extraction.main
from python_metal_fe.feature_extraction.main import MetaFeatureExtraction
import python_metal_fe.meta_get_features
from python_metal_fe.meta_get_features import EncoderDecoderNetwork
from python_metal_fe.model_tuning import model_tune
import sys
from tqdm import tqdm
import tqdm.std
from tqdm.std import TMonitor
import unittest


class Meta_get_featuresTest(unittest.TestCase):

    @patch.object(MetaFeatureExtraction, 'load_model')

    @patch.object(ntpath, 'join')
    @patch.object(MetaFeatureExtraction, 'gather_meta_features')
    @patch.object(numpy.lib.npyio, 'save')
    @patch.object(MetaFeatureExtraction, 'gather_random_addresses')
    @patch.object(TMonitor, '__init__')
    @patch.object(TMonitor, '__new__')
    @patch.object(MetaFeatureExtraction, '__init__')
    @patch.object(genericpath, 'exists')
    def test_main(self, mock_exists, mock___init__, mock___iter__, mock_build_encoder, mock_load_weights, mock___new__, mock_gather_random_addresses, mock_save, mock_gather_meta_features, mock_join, mock_update_encoder_weights, mock_load_model, mock_build_decoder):
        mock_exists.return_value = True
        mock___init__.return_value = None
        mock_build_encoder.return_value = None
        mock_load_weights.return_value = None
        mock___init__.return_value = None
        mock___new__.return_value = tqdm()
        mock___init__.return_value = None
        mock_gather_random_addresses.return_value = None
        mock_save.return_value = None
        mock_gather_meta_features.return_value = None
        mock_join.return_value = 'metafeature_extraction_result\\meta_regressor_features_Task04_Hippocampus_VGG16.npy'
        mock_update_encoder_weights.return_value = None
        mock_load_model.return_value = None
        mock_build_decoder.return_value = None
        self.assertEqual(
            python_metal_fe.meta_get_features.main(args=[]),
            None
        )


    @patch.object(ArgumentParser, 'parse_args')
    @patch.object(ArgumentParser, 'add_argument')
    @patch.object(ArgumentParser, '__init__')
    def test_parse_args(self, mock___init__, mock_add_argument, mock_parse_args):
        mock___init__.return_value = None
        mock_add_argument.return_value = _StoreAction()
        mock_parse_args.return_value = Namespace()
        self.assertIsInstance(
            python_metal_fe.meta_get_features.parse_args(args=[]),
            argparse.Namespace
        )





if __name__ == "__main__":
    unittest.main()