import pytest
import sys
sys.path.append('.')
import learn
import numpy as np

def test_memory():
    memory = learn.Memory(300)    
    for i in range(300):
        memory.add(i, None, None, None)
        assert memory.len() == i + 1
    memory.add(1, None, None, None)
    assert memory.len() == 300
    sample = memory.sample(300)
    for s in sample:
        assert s != 0

# stab
class ModelStab:
    def __init__(self, param):
        self.param = param
    def predict(self, state):
        return [[state, state + self.param]]

class QNetworkStab:
    def __init__(self, param):
        self.model = ModelStab(param)

def test_get_td_error():
    memory = learn.PERMemory(300)
    state = 0
    action = 0
    reward = 0
    next_state = 1
    experience = (state, action, reward, next_state)
    main_qn = QNetworkStab(1)
    target_qn = QNetworkStab(-1)
    gamma = 0.5
    next_action = np.argmax(main_qn.model.predict(next_state)[0])
    target = reward + gamma * target_qn.model.predict(next_state)[0][next_action]
    answer = target - main_qn.model.predict(state)[0][action]
    assert memory.get_td_error(experience, gamma, main_qn, target_qn) == answer