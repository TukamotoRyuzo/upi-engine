import pytest
import sys
sys.path.append('.')
import upi
import numpy as np

def test_field(zero_field):
    field = upi.Field()
    assert field.X_MAX == 6
    assert field.Y_MAX == 13
    assert np.array_equal(field.field, zero_field)

def compare_pfen(pfen, answer):
    field = upi.Field()
    field.init_from_pfen(pfen)
    return np.array_equal(field.field, answer)

def test_init_from_pfen_empty(zero_field):
    pfen = '//////'
    assert compare_pfen(pfen, zero_field)

def test_init_from_pfen_ry(zero_field):
    pfen = 'ry//////'
    answer = zero_field
    answer[0, 0] = upi.Puyo.RED
    answer[0, 1] = upi.Puyo.YELLOW
    assert compare_pfen(pfen, answer)

def test_init_from_pfen_kenny(kenny):
    e = upi.Puyo.EMPTY
    g = upi.Puyo.GREEN
    b = upi.Puyo.BLUE
    p = upi.Puyo.PURPLE
    y = upi.Puyo.YELLOW
    answer = np.array([
        [p, p, p, y, p, p, p, g, p, y, p, y, e],
        [y, y, y, g, y, g, g, b, g, b, b, b, e],
        [g, g, g, b, y, b, b, y, b, y, y, b, y],
        [b, b, b, p, b, y, y, p, y, p, p, y, p],
        [p, p, p, b, y, p, p, g, p, g, g, p, g],
        [y, y, y, b, b, g, g, b, b, b, g, b, g]
    ])
    assert compare_pfen(kenny, answer)

# 同じFieldインスタンスを用いて2回連続でinit_from_pfenを呼び出しても正しく初期化されることのテスト。
def test_init_from_pfen_continuous(zero_field):
    field = upi.Field()
    pfen = 'ry//////'
    answer = zero_field.copy()
    answer[0, 0] = upi.Puyo.RED
    answer[0, 1] = upi.Puyo.YELLOW
    field.init_from_pfen(pfen)
    assert np.array_equal(field.field, answer)
    pfen = '//////'
    field.init_from_pfen(pfen)
    assert np.array_equal(field.field, zero_field)

# http://www.puyop.com/s/BizsHHBizBjGsHjBjGsGGsGIzzytrIjIIjIIjI
def test_delete_puyo(zero_field, kenny):
    field = upi.Field()
    field.init_from_pfen(kenny)
    chain, score = field.delete_puyo()
    assert np.array_equal(field.field, zero_field)
    assert chain == 19
    assert score == 175080

def test_is_empty(kenny):
    field = upi.Field()
    field.init_from_pfen('//////')
    assert field.is_empty()
    field.init_from_pfen(kenny)
    assert not field.is_empty()

def test_is_death(kenny):
    field = upi.Field()
    field.init_from_pfen('//////')
    assert not field.is_death()
    field.init_from_pfen(kenny)
    assert field.is_death()

def test_pretty(kenny):
    field = upi.Field()
    field.init_from_pfen(kenny)
    pretty_string = field.pretty()
    answer = 'eeypgg\r\n------\r\nybbypb\r\npbypgg\r\nybypgb\r\npgbypb\r\ngbypgb\r\npgbypg\r\npgbypg\r\npyybyb\r\nygbpbb\r\npygbpy\r\npygbpy\r\npygbpy'
    assert pretty_string == answer

def test_floors():
    field = upi.Field()
    pfen = 'ry//////'
    field.init_from_pfen(pfen)
    floors = field.floors()
    assert floors == [2, 0, 0, 0, 0, 0]

def test_floors2():
    field = upi.Field()
    pfen = 'rpyrbppyrrbby/yprbryb/////'
    field.init_from_pfen(pfen)
    floors = field.floors()
    assert floors == [13, 7, 0, 0, 0, 0]