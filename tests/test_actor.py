import pytest
import sys
sys.path.append('.')
import learn
import numpy as np

# stab
class ModelStab:
    def __init__(self):
        self.called = False
    def predict(self, state):
        self.called = True
        return [list(range(22))]

class QNetworkStab:
    action_size = 22
    def __init__(self):
        self.model = ModelStab()

def test_actor():
    actor = learn.Actor()
    qn = QNetworkStab()

    # ランダムな行動を取る
    action = actor.get_action(0, -0.1, qn)
    assert not qn.model.called

    # 必ず最適行動を取る
    action = actor.get_action(0, -1.9, qn)
    assert qn.model.called
    assert action == 21
    
