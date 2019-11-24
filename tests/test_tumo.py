import pytest
import sys
sys.path.append('.')
import upi

def test_tumo():
    tumo = upi.Tumo(upi.Puyo.RED, upi.Puyo.YELLOW)
    assert tumo.pivot == upi.Puyo.RED
    assert tumo.child == upi.Puyo.YELLOW
