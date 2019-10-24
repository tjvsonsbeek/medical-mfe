# -*- coding: utf-8 -*-

import pytest
from python_metal_fe.encoder-decoder_networks import EncoderDecoder

__author__ = "tjvsonsbeek"
__copyright__ = "tjvsonsbeek"
__license__ = "mit"


def test_enc_dec_init():
    enc_dec_vgg = EncoderDecoder('VGG16')
    enc_dec_resnet = EncoderDecoder('ResNet50')
    enc_dec_mbnet = EncoderDecoder('MobilenetV1')
    with pytest.raises(AssertionError):
        EncoderDecoder('somethingelse')

    assert fib(1) == 1
    assert fib(2) == 1
    assert fib(7) == 13
    with pytest.raises(AssertionError):
        fib(-10)
