import pytest
import sys
sys.path.append('.')
import upi

def test_to_puyo():
    assert upi.Puyo.EMPTY == upi.Puyo.to_puyo('e')
    assert upi.Puyo.RED == upi.Puyo.to_puyo('r')
    assert upi.Puyo.GREEN == upi.Puyo.to_puyo('g')
    assert upi.Puyo.BLUE == upi.Puyo.to_puyo('b')
    assert upi.Puyo.PURPLE == upi.Puyo.to_puyo('p')
    assert upi.Puyo.YELLOW == upi.Puyo.to_puyo('y')
    assert upi.Puyo.OJAMA == upi.Puyo.to_puyo('o')

def test_to_str():
    assert upi.Puyo.EMPTY.to_str() == 'e'
    assert upi.Puyo.RED.to_str() == 'r'
    assert upi.Puyo.GREEN.to_str() == 'g'
    assert upi.Puyo.BLUE.to_str() == 'b'
    assert upi.Puyo.PURPLE.to_str() == 'p'
    assert upi.Puyo.YELLOW.to_str() == 'y'
    assert upi.Puyo.OJAMA.to_str() == 'o'