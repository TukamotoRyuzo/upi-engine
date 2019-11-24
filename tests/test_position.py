import pytest
import sys
sys.path.append('.')
import upi
import numpy as np
import random

def test_position():
    pos = upi.Position()
    assert pos.tumo_index == 0
    assert pos.ojama_index == 0
    assert pos.fall_bonus == 0
    assert not pos.all_clear_flag

def test_get_move_range():
    pos = upi.Position()
    floors = pos.field.floors()
    bx, ex = upi.get_move_range(floors)
    assert bx == 0
    assert ex == 5

def test_generate_moves(tumo_pool):
    pos = upi.Position()
    moves = upi.generate_moves(pos, tumo_pool)
    assert len(moves) == 22

def test_generate_moves_samecolor(tumo_pool_all_red):
    pos = upi.Position()
    moves = upi.generate_moves(pos, tumo_pool_all_red)
    assert len(moves) == 11

def test_do_move():
    pos = upi.Position()
    com = upi.PositionsCommonInfo()
    moves = upi.generate_moves(pos, com.tumo_pool)
    pos.do_move(moves[0], com)
    assert pos.field.get_puyo(0, 0) == com.tumo_pool[0].pivot
    assert pos.field.get_puyo(0, 1) == com.tumo_pool[0].child
    assert pos.tumo_index == 1

def test_fall_ojama():
    pos = upi.Position()
    com = upi.PositionsCommonInfo()
    com.future_ojama.fixed_ojama = 25
    pos.fall_ojama(com)
    pos.ojama_index = 24
    for x in range(6):
        for y in range(4):
            assert pos.field.get_puyo(x, y) == upi.Puyo.OJAMA
    v = list(range(6))
    for x in range(6):
        t = com.tumo_pool[pos.ojama_index]
        r = (t.pivot.value + t.child.value) % 6
        v[x], v[r] = v[r], v[x]
        pos.ojama_index = pos.ojama_index + 1 % 128
    for x in range(6):
        if x == v[0]:
            assert pos.field.get_puyo(x, 4) == upi.Puyo.OJAMA
        else:
            assert pos.field.get_puyo(x, 4) == upi.Puyo.EMPTY

