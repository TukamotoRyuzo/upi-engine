import pytest
import sys
sys.path.append('.')
import learn
import numpy as np
import upi

# stab
class ModelStab:
    def __init__(self):
        self.called = False
    def predict(self, state):
        self.called = True
        return [list(range(22))]

class QNetworkStab:
    ACTION_SIZE = 22
    def __init__(self):
        self.model = ModelStab()

# ニューラルネットワークに入力する形になっていることを確かめるテスト。
def test_state():
    stab = QNetworkStab()
    env = learn.BattleEnvironment(stab)
    env.player.positions[0].field.set_puyo(0, 0, upi.Puyo.RED)
    env.player.positions[0].field.set_puyo(0, 1, upi.Puyo.RED)
    env.player.positions[0].tumo_index = 1
    env.player.positions[1].field.set_puyo(0, 0, upi.Puyo.BLUE)
    env.player.positions[1].field.set_puyo(0, 1, upi.Puyo.BLUE)
    env.player.positions[1].tumo_index = 2
    state = env.get_state()
    assert len(state) == 9
    assert state[0].shape == (1, 6, 13, 6)
    assert state[1].shape == (1, 2, 5)
    assert state[2].shape == (1, 2, 5)
    assert state[3].shape == (1, 1)
    assert state[4].shape == (1, 6, 13, 6)
    assert state[5].shape == (1, 2, 5)
    assert state[6].shape == (1, 2, 5)
    assert state[7].shape == (1, 1)
    assert state[8].shape == (1, 3)

def test_step():
    stab = QNetworkStab()
    env = learn.BattleEnvironment(stab)
    action = 0
    next_state, reward, done = env.step(action)
    
