import pytest
import sys
sys.path.append('.')
import upi
import numpy as np
import random

def test_search(zero_field):
    upi_player = upi.UpiPlayer()    
    move = upi.search(upi_player.positions[0], upi_player.positions[1], upi_player.common_info, 1)
    assert move.to_upi() == '1a1b'
    assert np.array_equal(upi_player.positions[0].field.field, zero_field)

def test_search_death(zero_field):
    upi_player = upi.UpiPlayer()    
    upi_player.positions[0].field.init_from_pfen('//oooooooooooo////')
    move = upi.search(upi_player.positions[0], upi_player.positions[1], upi_player.common_info, 1)
    assert move.to_upi() == upi.Move.none().to_upi()