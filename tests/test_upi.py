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

def test_tumo(tumo_pool_string):
    upi_player = upi.UpiPlayer()
    upi_player.tumo(tumo_pool_string)
    for i, t in enumerate(tumo_pool_string):
        assert upi_player.common_info.tumo_pool[i].pivot == upi.Puyo.to_puyo(t[0])
        assert upi_player.common_info.tumo_pool[i].child == upi.Puyo.to_puyo(t[1])
    
def test_position():
    upi_player = upi.UpiPlayer()
    cmd = ['ryrb//////', '2', '//////', '3', '-1', '4', '250']
    upi_player.position(cmd)
    field = upi.Field()
    field.init_from_pfen(cmd[0])    
    assert np.array_equal(upi_player.positions[0].field.field, field.field)
    assert upi_player.positions[0].tumo_index == int(cmd[1])
    field.init_from_pfen(cmd[2])
    assert np.array_equal(upi_player.positions[1].field.field, field.field)
    assert upi_player.positions[1].tumo_index == int(cmd[3])
    assert upi_player.common_info.future_ojama.fixed_ojama == int(cmd[4])
    assert upi_player.common_info.future_ojama.unfixed_ojama == int(cmd[5])
    assert upi_player.common_info.future_ojama.time_until_fall_ojama == int(cmd[6])

def test_rule():
    upi_player = upi.UpiPlayer()
    cmd = ['falltime', '22', 'chaintime', '630', 'settime', '5', 'nexttime', '347', 'autodroptime', '52340']
    upi_player.rule(cmd)
    assert upi_player.common_info.rule.fall_time == int(cmd[1])
    assert upi_player.common_info.rule.chain_time == int(cmd[3])
    assert upi_player.common_info.rule.set_time == int(cmd[5])
    assert upi_player.common_info.rule.next_time == int(cmd[7])
    assert upi_player.common_info.rule.autodrop_time == int(cmd[9])

def test_go(capsys, tumo_pool_string):
    upi_player = upi.UpiPlayer()
    upi_player.tumo(tumo_pool_string)
    upi_player.position(['//////', '0', '//////', '0', '0', '0', '0'])
    upi_player.go()
    out, _ = capsys.readouterr()
    assert out == ('bestmove 1a1b\n')

def test_various_error(capsys):
    tumo = 'gg rp pg rp pb bb rr rb rg bg gr br gb gr br pr bb bb bb rg bg pp pr gr pb rg gp gr pg gg bp pp bb gr pp bb bp br rr pp rb pr bg bp gp rg bb gb pg bb bb bp pr pg gp bp pb rp pp pr rb gp pp pb br br rb gg pr pp rb pg rb gr gr gb pb rb gg rr pb pb bg pg gb pp gg gb gp rg pp bg gr rp bb bb gb rp gp bp gg gp gp pr pb rb gb pp rg br bb bp rr gp gr gr rr bg pr pb gb gp bp br gr rr pr pb'
    rule = 'falltime 2 chaintime 60 settime 15 nexttime 7 autodroptime 50'
    position = 'grpprbgrbggr/gpbrgrbbgrpp/gpprb///pbp/ 41 ggobbob/pgppbrrr/rproopp/rrooobbgbprr/gbobbr/rrrpobbg/ 23 0 296 574'
    upi_player = upi.UpiPlayer()
    upi_player.tumo(tumo.split(' '))
    upi_player.rule(rule.split(' '))
    upi_player.position(position.split(' '))
    upi_player.go()
    # out, _ = capsys.readouterr()
    # assert out == ('bestmove 1l1m\n')