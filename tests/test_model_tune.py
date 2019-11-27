# -*- coding: utf-8 -*-

import pytest
import os
from python_metal_fe.src.python_metal_fe.encoder_decoder_networks import EncoderDecoderNetwork as EncoderDecoder

__author__ = "tjvsonsbeek"
__copyright__ = "tjvsonsbeek"
__license__ = "mit"


def test_fib():
    assert fib(1) == 1
    assert fib(2) == 1
    assert fib(7) == 13
    with pytest.raises(AssertionError):
        fib(-10)
