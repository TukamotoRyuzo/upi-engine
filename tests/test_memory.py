import pytest
import sys
sys.path.append('.')
import learn
import numpy as np

def test_memory():
    memory = learn.Memory(300)    
    for i in range(300):
        memory.add(i)
        assert memory.len() == i + 1
    memory.add(1)
    assert memory.len() == 300
    sample = memory.sample(300)
    for s in sample:
        assert s != 0        