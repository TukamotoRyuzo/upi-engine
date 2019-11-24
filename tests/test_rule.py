import pytest
import sys
sys.path.append('.')
import upi

def test_rule():
    rule = upi.Rule()
    assert rule.chain_time == 60
    assert rule.next_time == 7
    assert rule.set_time == 15
    assert rule.fall_time == 2
    assert rule.autodrop_time == 50
