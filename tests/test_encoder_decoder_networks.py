# -*- coding: utf-8 -*-

import pytest
import os
from python_metal_fe.src.python_metal_fe.encoder_decoder_networks import EncoderDecoderNetwork as EncoderDecoder

__author__ = "tjvsonsbeek"
__copyright__ = "tjvsonsbeek"
__license__ = "mit"


def test_enc_dec_init():

    enc_dec_vgg = EncoderDecoder('VGG16')

    enc_dec_resnet = EncoderDecoder('ResNet50')

    enc_dec_mbnet = EncoderDecoder('MobilenetV1')

    enc_dec_else = EncoderDecoder('somethingelse')

def test_enc_dec_attribute():
    enc_dec_vgg = EncoderDecoder('VGG16')

    enc_dec_resnet = EncoderDecoder('ResNet50')
    enc_dec_else = EncoderDecoder('somethingelse')
    with pytest.raises(AttributeError):
        enc_dec_resnet.build_decoder( )
    with pytest.raises(AttributeError):
        enc_dec_vgg.build_decoder( )
    with pytest.raises(AttributeError):
        enc_dec_else.build_decoder( )

def test_enc_dec_complete():
    enc_dec_vgg = EncoderDecoder('VGG16')
    enc_dec_mbnet = EncoderDecoder('MobileNetV1')
    enc_dec_resnet = EncoderDecoder('ResNet50')
    enc_dec_else = EncoderDecoder('somethingelse')
    enc_dec_vgg.build_encoder( )
    enc_dec_resnet.build_encoder( )
    enc_dec_mbnet.build_encoder( )
    with pytest.raises(AssertionError):
        enc_dec_else.build_encoder( )

    enc_dec_vgg.build_decoder( )
    enc_dec_resnet.build_decoder( )
    enc_dec_mbnet.build_decoder( )


    enc_dec_vgg.build_classifier( )
    enc_dec_resnet.build_classifier( )
    enc_dec_mbnet.build_classifier( )

# def test_enc_dec_paths():
#     enc_dec_vgg = EncoderDecoder('ENCODER')
#     assert os.path.exists(self.model_tune_weights_path)
    # assert os.path.exists(LOCAL_INSTALL_DIR)
    # assert os.path.exists(LOCAL_INSTALL_DIR)
    # assert os.path.exists(LOCAL_INSTALL_DIR)
