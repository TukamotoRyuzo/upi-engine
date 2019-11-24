import pytest
import sys
sys.path.append('.')
import upi
import random
import numpy as np

def test_upi(capsys):
    upi_player = upi.UpiPlayer()
    upi_player.upi()
    out, _ = capsys.readouterr()
    assert out == (
        'id name sample_engine1.0\n'
        'id author Ryuzo Tukamoto\n'
        'upiok\n'
    )

def test_isready(capsys):
    upi_player = upi.UpiPlayer()
    upi_player.isready()
    out, _ = capsys.readouterr()
    assert out == ('readyok\n')

def test_tumo():
    upi_player = upi.UpiPlayer()
    tumo = ["rgbp"[random.randrange(4)] + "rgbp"[random.randrange(4)] for i in range(128)]
    upi_player.tumo(tumo)
    for i, t in enumerate(tumo):
        assert upi_player._tumo_pool[i].pivot == upi.Puyo.to_puyo(t[0])
        assert upi_player._tumo_pool[i].child == upi.Puyo.to_puyo(t[1])
    
def test_position():
    upi_player = upi.UpiPlayer()
    cmd = ['ryrb//////', '2', '//////', '3', '-1', '4', '250']
    upi_player.position(cmd)
    field = upi.Field()
    field.init_from_pfen(cmd[0])    
    assert np.array_equal(upi_player._position[0].field.field, field.field)
    assert upi_player._position[0].tumo_index == int(cmd[1])
    field.init_from_pfen(cmd[2])
    assert np.array_equal(upi_player._position[1].field.field, field.field)
    assert upi_player._position[1].tumo_index == int(cmd[3])
    assert upi_player._fixed_ojama == int(cmd[4])
    assert upi_player._unfixed_ojama == int(cmd[5])
    assert upi_player._time_until_fall_ojama == int(cmd[6])

def test_rule():
    upi_player = upi.UpiPlayer()
    cmd = ['falltime', '22', 'chaintime', '630', 'settime', '5', 'nexttime', '347', 'autodroptime', '52340']
    upi_player.rule(cmd)
    assert upi_player._rule.fall_time == int(cmd[1])
    assert upi_player._rule.chain_time == int(cmd[3])
    assert upi_player._rule.set_time == int(cmd[5])
    assert upi_player._rule.next_time == int(cmd[7])
    assert upi_player._rule.autodrop_time == int(cmd[9])
