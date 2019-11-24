import pytest
import sys
sys.path.append('.')
import upi
import numpy as np
import random

@pytest.fixture
def tumo_pool():
    return [upi.Tumo(upi.Puyo.RED, upi.Puyo.GREEN) for i in range(128)]

@pytest.fixture
def tumo_pool_all_red():
    return [upi.Tumo(upi.Puyo.RED, upi.Puyo.RED) for i in range(128)]

@pytest.fixture
def tumo_pool_string():
    return ["rgbp"[random.randrange(4)] + "rgbp"[random.randrange(4)] for i in range(128)]

@pytest.fixture
def zero_field():
    return np.full((upi.Field.X_MAX, upi.Field.Y_MAX), upi.Puyo.EMPTY)

@pytest.fixture
def kenny():
    return 'pppypppgpypy/yyygyggbgbbb/gggbybbybyyby/bbbpbyypyppyp/pppbyppgpggpg/yyybbggbbbgbg/'