import pytest
import sys
sys.path.append('.')
import upi
import numpy as np
import random

def test_move():
    pivot_sq = (0, 0)
    child_sq = (0, 1)
    move = upi.Move(pivot_sq, child_sq, True)
    assert move.pivot_sq == pivot_sq
    assert move.child_sq == child_sq
    assert move.is_tigiri == True

def test_to_upi():
    assert upi.Move((0, 0), (0, 1)).to_upi() == '1a1b'
    assert upi.Move((4, 9), (5, 10)).to_upi() == '5j6k'