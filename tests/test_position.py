import pytest
import sys
sys.path.append('.')
import upi
import numpy as np
import random

def test_get_move_range():
    pos = upi.Position()
    floors = pos.field.floors()
    bx, ex = upi.get_move_range(floors)
    assert bx == 0
    assert ex == 5

def test_generate_moves():
    pos = upi.Position()
    tumo = upi.Tumo(upi.Puyo.YELLOW, upi.Puyo.GREEN)
    moves = upi.generate_moves(pos, tumo)
    assert len(moves) == 22

def test_generate_moves_samecolor():
    pos = upi.Position()
    tumo = upi.Tumo(upi.Puyo.YELLOW, upi.Puyo.YELLOW)
    moves = upi.generate_moves(pos, tumo)
    assert len(moves) == 11
