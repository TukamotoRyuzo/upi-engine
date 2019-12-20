import pytest
import sys
sys.path.append('.')
import learn
import numpy as np
import upi

# ニューラルネットワークに入力する形になっていることを確かめるテスト。
def test_state():
    env = learn.TokotonEnvironment()
    state = env.get_state()
    tumo = env.player.common_info.tumo_pool[0]
    curr_tumo_answer = np.zeros((1, 2, 5))
    curr_tumo_answer[0, 0, tumo.pivot.value - 1] = 1
    curr_tumo_answer[0, 1, tumo.child.value - 1] = 1
    tumo = env.player.common_info.tumo_pool[1]
    next_tumo_answer = np.zeros((1, 2, 5))
    next_tumo_answer[0, 0, tumo.pivot.value - 1] = 1
    next_tumo_answer[0, 1, tumo.child.value - 1] = 1
    assert np.array_equal(state[0], np.zeros((1, 6, 13, 5)))
    assert np.array_equal(state[1], curr_tumo_answer)
    assert np.array_equal(state[2], next_tumo_answer)

def test_action_to_move():
    env = learn.TokotonEnvironment()
    assert env.action_to_move( 0).to_upi() == '1a1b'
    assert env.action_to_move( 1).to_upi() == '2a2b'
    assert env.action_to_move( 2).to_upi() == '3a3b'
    assert env.action_to_move( 3).to_upi() == '4a4b'
    assert env.action_to_move( 4).to_upi() == '5a5b'
    assert env.action_to_move( 5).to_upi() == '6a6b'
    assert env.action_to_move( 6).to_upi() == '1b1a'
    assert env.action_to_move( 7).to_upi() == '2b2a'
    assert env.action_to_move( 8).to_upi() == '3b3a'
    assert env.action_to_move( 9).to_upi() == '4b4a'
    assert env.action_to_move(10).to_upi() == '5b5a'
    assert env.action_to_move(11).to_upi() == '6b6a'
    assert env.action_to_move(12).to_upi() == '1a2a'
    assert env.action_to_move(13).to_upi() == '2a3a'
    assert env.action_to_move(14).to_upi() == '3a4a'
    assert env.action_to_move(15).to_upi() == '4a5a'
    assert env.action_to_move(16).to_upi() == '5a6a'
    assert env.action_to_move(17).to_upi() == '2a1a'
    assert env.action_to_move(18).to_upi() == '3a2a'
    assert env.action_to_move(19).to_upi() == '4a3a'
    assert env.action_to_move(20).to_upi() == '5a4a'
    assert env.action_to_move(21).to_upi() == '6a5a'

def test_action_to_move_samecolor_tumo():
    env = learn.TokotonEnvironment()
    env.player.common_info.tumo_pool = [upi.Tumo(upi.Puyo.RED, upi.Puyo.RED) for i in range(128)]
    assert env.action_to_move( 0).to_upi() == '1a1b'
    assert env.action_to_move( 1).to_upi() == '2a2b'
    assert env.action_to_move( 2).to_upi() == '3a3b'
    assert env.action_to_move( 3).to_upi() == '4a4b'
    assert env.action_to_move( 4).to_upi() == '5a5b'
    assert env.action_to_move( 5).to_upi() == '6a6b'
    assert env.action_to_move( 6).to_upi() == '1a1b'
    assert env.action_to_move( 7).to_upi() == '2a2b'
    assert env.action_to_move( 8).to_upi() == '3a3b'
    assert env.action_to_move( 9).to_upi() == '4a4b'
    assert env.action_to_move(10).to_upi() == '5a5b'
    assert env.action_to_move(11).to_upi() == '6a6b'
    assert env.action_to_move(12).to_upi() == '1a2a'
    assert env.action_to_move(13).to_upi() == '2a3a'
    assert env.action_to_move(14).to_upi() == '3a4a'
    assert env.action_to_move(15).to_upi() == '4a5a'
    assert env.action_to_move(16).to_upi() == '5a6a'
    assert env.action_to_move(17).to_upi() == '1a2a'
    assert env.action_to_move(18).to_upi() == '2a3a'
    assert env.action_to_move(19).to_upi() == '3a4a'
    assert env.action_to_move(20).to_upi() == '4a5a'
    assert env.action_to_move(21).to_upi() == '5a6a'

def test_action_to_move():
    env = learn.TokotonEnvironment()
    env.player.positions[0].field.init_from_pfen('/oooooooooooo//oooooooooooo///')
    assert env.action_to_move( 0).is_none()
    assert env.action_to_move( 1).is_none()
    assert env.action_to_move( 2).to_upi() == '3a3b'
    assert env.action_to_move( 3).is_none()
    assert env.action_to_move( 4).is_none()
    assert env.action_to_move( 5).is_none()
    assert env.action_to_move( 6).is_none()
    assert env.action_to_move( 7).is_none()
    assert env.action_to_move( 8).to_upi() == '3b3a'
    assert env.action_to_move( 9).is_none()
    assert env.action_to_move(10).is_none()
    assert env.action_to_move(11).is_none()
    assert env.action_to_move(12).is_none()
    assert env.action_to_move(13).is_none()
    assert env.action_to_move(14).is_none()
    assert env.action_to_move(15).is_none()
    assert env.action_to_move(16).is_none()
    assert env.action_to_move(17).is_none()
    assert env.action_to_move(18).is_none()
    assert env.action_to_move(19).is_none()
    assert env.action_to_move(20).is_none()
    assert env.action_to_move(21).is_none()

def test_reward():
    env = learn.TokotonEnvironment()    
    assert env.get_reward(False) == 0
    env.player.common_info.future_ojama.fixed_ojama = -99999
    assert env.get_reward(False) == 0
    assert env.get_reward(True) == 1

def test_step():
    env = learn.TokotonEnvironment()
    state, reward, done = env.step(0)
    assert reward == 0
    assert not done
    tumo = env.player.common_info.tumo_pool[0]
    field_answer = np.zeros((1, 6, 13, 5))
    field_answer[0, 0, 0, tumo.pivot.value - 1] = 1
    field_answer[0, 0, 1, tumo.child.value - 1] = 1
    tumo = env.player.common_info.tumo_pool[1]
    curr_tumo_answer = np.zeros((1, 2, 5))
    curr_tumo_answer[0, 0, tumo.pivot.value - 1] = 1
    curr_tumo_answer[0, 1, tumo.child.value - 1] = 1
    tumo = env.player.common_info.tumo_pool[2]
    next_tumo_answer = np.zeros((1, 2, 5))
    next_tumo_answer[0, 0, tumo.pivot.value - 1] = 1
    next_tumo_answer[0, 1, tumo.child.value - 1] = 1
    assert np.array_equal(state[0], field_answer)
    assert np.array_equal(state[1], curr_tumo_answer)
    assert np.array_equal(state[2], next_tumo_answer)
    assert state is not None

def test_step_death():
    env = learn.TokotonEnvironment()
    env.player.positions[0].field.init_from_pfen('//ooooooooooo////')
    state, reward, done = env.step(2)
    assert reward == -1
    assert done
    assert state is None

def test_step_done():
    env = learn.TokotonEnvironment()
    env.player.positions[0].field.init_from_pfen('//////')
    env.player.common_info.future_ojama.fixed_ojama = -99999
    state, reward, done = env.step(2)
    assert reward == 1
    assert done
    assert state is None

def test_step_illegalmove_done():
    env = learn.TokotonEnvironment()
    env.player.positions[0].field.init_from_pfen('/oooooooooooo//oooooooooooo///')
    state, reward, done = env.step(0)
    assert reward == -1
    assert done
    assert state is None

def test_reset():
    env = learn.TokotonEnvironment()
    env.player.positions[0].field.init_from_pfen('g/r/g/r/g/r/')
    env.player.positions[0].tumo_index = 20
    env.reset()
    assert env.player.positions[0].field.is_empty()
    assert env.player.positions[0].tumo_index == 0
