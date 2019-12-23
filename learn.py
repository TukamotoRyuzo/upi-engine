from keras.layers import Activation, Conv2D, Dense, Input, Flatten, MaxPooling2D, concatenate, Lambda
from keras.optimizers import Adam, RMSprop
from keras.models import Sequential, Model
from keras.utils import plot_model
from keras.callbacks import TensorBoard
from collections import deque
from datetime import datetime
from keras import backend as K
import tensorflow as tf
import upi
import numpy as np
import os
import argparse
from abc import ABC, abstractmethod

# 損失関数にhuber関数を使用します 
# 参考: https://github.com/jaara/AI-blog/blob/master/CartPole-DQN.py
def huberloss(y_true, y_pred):
    err = y_true - y_pred
    cond = K.abs(err) < 1.0
    L2 = 0.5 * K.square(err)
    L1 = (K.abs(err) - 0.5)
    loss = tf.where(cond, L2, L1)  # Keras does not cover where function in tensorflow :-(
    return K.mean(loss)


class QNetworkBase(ABC):
    """
    指し手22種類の中から最善手を選ぶためのニューラルネットワーク。
    時刻tでの状態sにおける行動aによって、この先どの程度の報酬Rがトータルでもらえるのかを返す。
    """    

    @abstractmethod
    def __init__(self, learning_rate=0.0001):
        """
        ネットワークのモデルを構築する。
        """  
        pass
    
    @abstractmethod
    def get_zero_states(self, batch_size):
        pass
    
    @abstractmethod
    def store_states(self, states, state, i):
        pass

    def make_teacher_label(self, batch_size, gamma, target_qn, mini_batch):
        """
        教師データを作成する。
        """
        states = self.get_zero_states(batch_size)
        targets = np.zeros((batch_size, PuyoGym.ACTION_SIZE))
        errors = np.zeros(batch_size)
        for i, (_, (state_b, action_b, reward_b, next_state_b)) in enumerate(mini_batch):
            self.store_states(states, state_b, i)
            target = reward_b
            if next_state_b is not None:
                # Double DQN: 今の重みwにより選ばれた行動を、直前の重みw'で評価する
                retmain_qs = self.model.predict(next_state_b)[0]
                next_action = np.argmax(retmain_qs)
                target = reward_b + gamma * target_qn.model.predict(next_state_b)[0][next_action]
            targets[i] = self.model.predict(state_b)[0]
            errors[i] = abs(targets[i][action_b] - target)
            targets[i][action_b] = target            
        return states, targets, errors
    
    def replay(self, memory, batch_size, gamma, target_qn):
        """
        重みの学習を行う。
        """
        mini_batch = memory.sample(batch_size)
        inputs, targets, errors = self.make_teacher_label(batch_size, gamma, target_qn, mini_batch)
        for i in range(batch_size):
            idx = mini_batch[i][0]
            memory.update(idx, errors[i])
        self.model.fit(inputs, targets, epochs=1, verbose=0)


class TokotonNetwork(QNetworkBase):
    def __init__(self, learning_rate=0.0001):        
        # 入力
        # 盤面, ツモ(現在のツモ、次のツモ)
        field = Input(shape=PuyoGym.FIELD_SHAPE, name='field')
        tumo_curr = Input(shape=PuyoGym.TUMO_SHAPE, name='tumo_curr')
        tumo_next = Input(shape=PuyoGym.TUMO_SHAPE, name='tumo_next')
        all_clear_flag = Input(shape=(1,), name='all_clear_flag')

        # ネットワークを定義する
        x = Conv2D(128, 5, activation='relu', padding='same')(field)
        x = Conv2D(128, 3, activation='relu', padding='same')(x)
        x = Flatten()(x)
        y = Flatten()(tumo_curr)
        z = Flatten()(tumo_next)
        all = concatenate([x, y, z, all_clear_flag])
        
        # dueling network
        # 状態価値
        v = Dense(64, activation='relu')(all)
        v = Dense(1)(v)

        # 行動価値
        adv = Dense(64, activation='relu')(all)
        adv = Dense(PuyoGym.ACTION_SIZE)(adv)
        
        y = concatenate([v,adv])
        output = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - tf.stop_gradient(K.mean(a[:,1:],keepdims=True)), output_shape=(PuyoGym.ACTION_SIZE,))(y)
        self.model = Model(inputs=[field, tumo_curr, tumo_next, all_clear_flag], outputs=output)
        self.model.compile(optimizer=RMSprop(lr=learning_rate), loss=huberloss)
    
    def get_zero_states(self, batch_size):
        field = np.zeros(PuyoGym.get_batch_field_shape(batch_size))
        tumo_curr = np.zeros(PuyoGym.get_batch_tumo_shape(batch_size))
        tumo_next = np.zeros(PuyoGym.get_batch_tumo_shape(batch_size))
        all_clear_flag = np.zeros((batch_size, 1))
        return [field, tumo_curr, tumo_next, all_clear_flag]
    
    def store_states(self, states, state, i):
        states[0][i] = state[0]
        states[1][i] = state[1]
        states[2][i] = state[2]
        states[3][i] = state[3]

class BattleNetwork(QNetworkBase):
    def __init__(self, learning_rate=0.0001):        
        # 入力
        # 盤面, ツモ(現在のツモ、次のツモ)
        field = [None] * 2
        tumo_curr = [None] * 2
        tumo_next = [None] * 2
        all_clear_flag = [None] * 2
        concat = [None] * 2
        for i in range(2):
            field[i] = Input(shape=PuyoGym.FIELD_SHAPE, name=f'field{i}')
            tumo_curr[i] = Input(shape=PuyoGym.TUMO_SHAPE, name=f'tumo_curr{i}')
            tumo_next[i] = Input(shape=PuyoGym.TUMO_SHAPE, name=f'tumo_next{i}')
            all_clear_flag[i] = Input(shape=PuyoGym.ALLCLEAR_SHAPE, name=f'all_clear_flag{i}')
            x = Conv2D(128, 5, activation='relu', padding='same')(field[i])
            x = Conv2D(128, 3, activation='relu', padding='same')(x)
            x = Flatten()(x)
            y = Flatten()(tumo_curr[i])
            z = Flatten()(tumo_next[i])
            concat[i] = concatenate([x, y, z, all_clear_flag[i]])
        ojama = Input(shape=PuyoGym.OJAMA_SHAPE, name='ojama')        
        all = concatenate([concat[0], concat[1], ojama])

        # dueling network
        # 状態価値
        v = Dense(64, activation='relu')(all)
        v = Dense(1)(v)

        # 行動価値
        adv = Dense(64, activation='relu')(all)
        adv = Dense(PuyoGym.ACTION_SIZE)(adv)
        
        y = concatenate([v,adv])
        outputs = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - tf.stop_gradient(K.mean(a[:,1:],keepdims=True)), output_shape=(PuyoGym.ACTION_SIZE,))(y)
        inputs = [field[0], tumo_curr[0], tumo_next[0], all_clear_flag[0], field[1], tumo_curr[1], tumo_next[1], all_clear_flag[1], ojama]
        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer=RMSprop(lr=learning_rate), loss=huberloss)
    
    def get_zero_states(self, batch_size):
        field0 = np.zeros(PuyoGym.get_batch_field_shape(batch_size))
        tumo_curr0 = np.zeros(PuyoGym.get_batch_tumo_shape(batch_size))
        tumo_next0 = np.zeros(PuyoGym.get_batch_tumo_shape(batch_size))
        all_clear_flag0 = np.zeros(PuyoGym.get_batch_allclear_shape(batch_size))
        field1 = np.zeros(PuyoGym.get_batch_field_shape(batch_size))
        tumo_curr1 = np.zeros(PuyoGym.get_batch_tumo_shape(batch_size))
        tumo_next1 = np.zeros(PuyoGym.get_batch_tumo_shape(batch_size))
        all_clear_flag1 = np.zeros(PuyoGym.get_batch_allclear_shape(batch_size))
        ojama = np.zeros(PuyoGym.get_batch_ojama_shape(batch_size))
        return [field0, tumo_curr0, tumo_next0, all_clear_flag0, field1, tumo_curr1, tumo_next1, all_clear_flag1, ojama]
    
    def store_states(self, states, state, i):
        states[0][i] = state[0]
        states[1][i] = state[1]
        states[2][i] = state[2]
        states[3][i] = state[3]
        states[4][i] = state[4]
        states[5][i] = state[5]
        states[6][i] = state[6]
        states[7][i] = state[7]
        states[8][i] = state[8]

class PuyoGym(ABC):
    """
    OpenAIGym形式の環境を作るためのクラス。
    """
    ACTION_SIZE = 22
    FIELD_SHAPE = (upi.Field.X_MAX, upi.Field.Y_MAX, 6)
    TUMO_SHAPE = (2, 5)
    OJAMA_SHAPE = (3,)
    ALLCLEAR_SHAPE = (1,)

    @staticmethod
    def get_batch_field_shape(batch_size):
        return (batch_size, PuyoGym.FIELD_SHAPE[0], PuyoGym.FIELD_SHAPE[1], PuyoGym.FIELD_SHAPE[2])
    
    @staticmethod
    def get_batch_tumo_shape(batch_size):
        return (batch_size, PuyoGym.TUMO_SHAPE[0], PuyoGym.TUMO_SHAPE[1])

    @staticmethod
    def get_batch_allclear_shape(batch_size):
        return (batch_size, PuyoGym.ALLCLEAR_SHAPE[0])

    @staticmethod
    def get_batch_ojama_shape(batch_size):
        return (batch_size, PuyoGym.OJAMA_SHAPE[0])

    def __init__(self):
        self.player = upi.UpiPlayer()

    def _get_state_impl(self, position, tumo_pool):
        """
        ぷよぷよのフィールド、現在のツモ、次のツモを、ニューラルネットワークに入力できる形に変換する。
        """
        # フィールド
        state_field = np.zeros(PuyoGym.FIELD_SHAPE)
        for x in range(upi.Field.X_MAX):
            for y in range(upi.Field.Y_MAX):
                puyo = position.field.get_puyo(x, y)
                if puyo != upi.Puyo.EMPTY:
                    state_field[x, y, puyo.value - 1] = 1
        # ツモ
        state_curr_tumo = np.zeros(PuyoGym.TUMO_SHAPE)
        state_next_tumo = np.zeros(PuyoGym.TUMO_SHAPE)
        curr_tumo = tumo_pool[(position.tumo_index + 0) % 128]
        next_tumo = tumo_pool[(position.tumo_index + 1) % 128]
        state_curr_tumo[0, curr_tumo.pivot.value - 1] = 1
        state_curr_tumo[1, curr_tumo.child.value - 1] = 1
        state_next_tumo[0, next_tumo.pivot.value - 1] = 1
        state_next_tumo[1, next_tumo.child.value - 1] = 1

        # 全消し
        state_all_clear = np.zeros(1)
        state_all_clear[0] = 1 if position.all_clear_flag else 0
        return [state_field[np.newaxis, :, :, :], state_curr_tumo[np.newaxis, :, :], state_next_tumo[np.newaxis, :, :], state_all_clear[np.newaxis]]

    def action_to_move(self, action, position, tumo_pool):
        """
        actionからmoveを求める。actionは以下の意味とする。
        0~5: 回転数0
        6~11: 回転数2
        12~16: 回転数1
        17~21: 回転数3
        """
        floors = position.field.floors()
        if action < 12:
            action_pivot_x = action % 6
            action_child_x = action_pivot_x
            if action < 6:
                action_pivot_y = floors[action_pivot_x]            
                action_child_y = action_pivot_y + 1
            else:
                action_child_y = floors[action_pivot_x]            
                action_pivot_y = action_child_y + 1
        elif action < 17:
            action_pivot_x = action - 12
            action_pivot_y = floors[action_pivot_x]
            action_child_x = action_pivot_x + 1
            action_child_y = floors[action_child_x]            
        else:
            action_pivot_x = action - 17 + 1
            action_pivot_y = floors[action_pivot_x]
            action_child_x = action_pivot_x - 1
            action_child_y = floors[action_child_x]
        move = upi.Move((action_pivot_x, action_pivot_y), (action_child_x, action_child_y), action_pivot_x != action_child_x and action_pivot_y != action_child_y)
        if move.pivot_sq[1] >= upi.Field.Y_MAX or move.child_sq[1] >= upi.Field.Y_MAX:
            return upi.Move.none()
        moves = upi.generate_moves(position, tumo_pool)        
        tumo = tumo_pool[position.tumo_index]

        # 反則手の判定をする。2列目と4列目が13段目まで埋まっているときに1列目におくなど。
        for m in moves:
            if move.to_upi() == m.to_upi():
                return m
            # ぞろ目なら、逆向きにおく手でもよい。
            if tumo.pivot == tumo.child:
                reversed_move = upi.Move(move.child_sq, move.pivot_sq, move.is_tigiri)
                if reversed_move.to_upi() == m.to_upi():
                    return m

        return upi.Move.none()

    def get_reward(self, done):
        if not done:
            return 0
        return 1 if self.success() else -1

    def reset(self):
        """
        環境を初期化する。
        """
        self.player.positions[0] = upi.Position()
        self.player.positions[1] = upi.Position()
        self.player.common_info = upi.PositionsCommonInfo()
        self.player.common_info.randomize_tumo()
        return self.get_state()
    
    def render(self):
        """
        環境を描画する。
        """
        self.player.positions[0].field.pretty_print()
        print('unfixed:', self.player.common_info.future_ojama.unfixed_ojama)
        print('fixed:', self.player.common_info.future_ojama.fixed_ojama)

    @abstractmethod
    def get_state(self):
        """
        ぷよぷよのフィールド、現在のツモ、次のツモを、ニューラルネットワークに入力できる形に変換する。
        """
        pass

    @abstractmethod
    def success(self):
        pass

    @abstractmethod
    def step(self, action):
        """
        actionを基にstateを次の状態に進める。
        """
        pass

class TokotonEnvironment(PuyoGym):
    """
    とこぷよ学習環境。
    """
    def __init__(self):
        super().__init__()
        self.goal = 30

    def set_goal(self, goal):
        self.goal = goal

    def get_state(self):
        return self._get_state_impl(self.player.positions[0], self.player.common_info.tumo_pool)

    def success(self):
        ojama = -(self.player.common_info.future_ojama.unfixed_ojama + self.player.common_info.future_ojama.fixed_ojama)
        return ojama >= self.goal

    def step(self, action):
        # 死んでいる局面でこのメソッドが呼び出されるはずがない。
        assert not self.player.positions[0].field.is_death()
        move = self.action_to_move(action, self.player.positions[0], self.player.common_info.tumo_pool)
        if not move.is_none():
            self.player.positions[0].do_move(move, self.player.common_info)
            state = self.get_state()            
            done = self.player.positions[0].field.is_death() or self.success()
        else:
            # おけないところに置こうとしたら負け。
            done = True
        reward = self.get_reward(done)
        if done:
            state = None
        return state, reward, done


class BattleEnvironment(PuyoGym):
    """
    対戦ぷよ学習環境。
    """
    def __init__(self, target_net):
        super().__init__()
        self.target_net = target_net

    def _get_state_battle(self, position0, position1, common_info):
        state_0 = self._get_state_impl(position0, common_info.tumo_pool)
        state_1 = self._get_state_impl(position1, common_info.tumo_pool)
        # おじゃまぷよ
        state_ojama = np.zeros(3)
        state_ojama[0] = common_info.future_ojama.fixed_ojama / 1000
        state_ojama[1] = common_info.future_ojama.unfixed_ojama / 1000
        state_ojama[2] = common_info.future_ojama.time_until_fall_ojama / 1000
        return state_0 + state_1 + [state_ojama[np.newaxis]]

    def get_state(self):
        return self._get_state_battle(self.player.positions[0], self.player.positions[1], self.player.common_info)

    def success(self):        
        return self.positions[1].field.is_death()

    def step(self, action):
        # 死んでいる局面でこのメソッドが呼び出されるはずがない。
        assert not self.player.positions[0].field.is_death()
        assert not self.player.positions[1].field.is_death()
        move = self.action_to_move(action, self.player.positions[0], self.player.common_info.tumo_pool)
        if not move.is_none():
            self.player.positions[0].do_move(move, self.player.common_info)

            # 死んだら負け
            if self.player.positions[0].field.is_death():
                print('my death')
                return None, -1, True

            # 相手の番
            if self.player.common_info.time < 0:
                self.player.common_info.inverse()
                while self.player.common_info.time >= 0:
                    state = self._get_state_battle(self.player.positions[1], self.player.positions[0], self.player.common_info)
                    enemy_action = Actor().get_action(state, 100000, self.target_net)
                    enemy_move = self.action_to_move(enemy_action, self.player.positions[1], self.player.common_info.tumo_pool)
                    if not move.is_none():
                        self.player.positions[1].do_move(enemy_move, self.player.common_info)

                        # 相手が死んだら勝ち
                        if self.player.positions[1].field.is_death():
                            print('your death')
                            return None, 1, True
                    else:
                        # 敵の反則手
                        print('your illegal move')
                        return None, 1, True

                # 手番入れ替え
                self.player.common_info.inverse()
        else:
            # 反則負け
            print('my illegal move')
            return None, -1, True

        state = self.get_state()
        return state, 0, False

    def render(self):
        """
        環境を描画する。
        """
        self.player.render()
        print('unfixed:', self.player.common_info.future_ojama.unfixed_ojama)
        print('fixed:', self.player.common_info.future_ojama.fixed_ojama)
        print('time_until_fall_ojama:', self.player.common_info.future_ojama.time_until_fall_ojama)
        print('time:', self.player.common_info.time)

class Memory:
    """
    経験再生用メモリ。
    """
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience, gamma, main_qn, target_qn):
        self.buffer.append(experience)

    def sample(self, batch_size):
        idx = np.random.choice(np.arange(self.len()), size=batch_size, replace=False)
        return [(i, self.buffer[ii]) for i, ii in enumerate(idx)]

    def len(self):
        return len(self.buffer)

    def update(self, idx, error):
        pass


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[data_idx])

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

class PERMemory:   # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.6

    def __init__(self, max_size):
        self.tree = SumTree(max_size)
        self.exp = 0

    def _get_priority(self, error):
        return (error + self.e) ** self.a

    def add(self, experience, gamma, main_qn, target_qn):
        error = abs(self.get_td_error(experience, gamma, main_qn, target_qn))
        p = self._get_priority(error)
        self.tree.add(p, experience) 
        if self.exp < self.tree.capacity:
            self.exp += 1
        
    def sample(self, batch_size):
        batch = []
        indices = []
        segment = self.tree.total() / batch_size
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            batch.append((idx, data))
        return batch

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)

    def len(self):
        return self.exp

    # TD誤差を取得
    @staticmethod
    def get_td_error(experience, gamma, main_qn, target_qn):
        (state, action, reward, next_state) = experience
        target = reward
        if next_state is not None:
            next_action = np.argmax(main_qn.model.predict(next_state)[0])
            target += gamma * target_qn.model.predict(next_state)[0][next_action]
        td_error = target - main_qn.model.predict(state)[0][action]
        return td_error


class Actor:
    def get_action(self, state, episode, main_qn):
        epsilon = 0.001 + 0.9 / (1.0 + episode)
        if epsilon <= np.random.uniform(0, 1):
            return np.argmax(main_qn.model.predict(state)[0])
        else:
            return np.random.choice(np.arange(PuyoGym.ACTION_SIZE))


class TensorBoardLogger():
    def __init__(self, log_dir):
        self.log_dir = log_dir        
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self.writer.set_as_default()
        self.step = tf.Variable(0.0, name='step')
        self.ojama = tf.Variable(0, name='ojama')
        
    def write(self, step, ojama, episode):
        self.step.assign(step / 100.0)
        self.ojama.assign(ojama)
        tf.summary.scalar('step', self.step, step=episode)
        tf.summary.scalar('ojama', self.ojama, step=episode)


def run(id, load_path):    
    num_episodes = 200000 # 総試行回数
    gamma = 0.9 # 割引係数
    memory_size = 1024
    batch_size = 4
    copy_target_freq = 10000
    learning_rate = 0.0001
    num_consecutive_iterations = 20
    total_reward_vec = np.zeros(num_consecutive_iterations)  # 各試行の報酬を格納

    # tensorboardによる可視化
    log_dir = f'.\\logs\\{id}-gamma{gamma}_memory{memory_size}_batch{batch_size}_update{copy_target_freq}_eta{learning_rate}'
    tensorboard = TensorBoardLogger(log_dir=log_dir)    
    save_weight_path = f'weights/{id}'

    # 環境作成    
    # main_qn = TokotonNetwork(learning_rate)
    # target_qn = TokotonNetwork(learning_rate)
    main_qn = BattleNetwork(learning_rate)
    target_qn = BattleNetwork(learning_rate)

    # env = TokotonEnvironment()
    env = BattleEnvironment(target_qn)
    #env.set_goal(30)

    if load_path != '':
        main_qn.model.load_weights(load_path)
        target_qn.model.load_weights(load_path)

    #memory = Memory(max_size=memory_size)
    memory = PERMemory(max_size=memory_size)
    actor = Actor()
    step = 0

    print('initialize memory')

    # メモリが満タンになるまでランダムプレイ
    while memory.len() < memory_size:
        env.reset()
        state = env.get_state()
        done = False
        while not done:
            #env.render()
            #a = input()                        
            action = np.random.randint(0, PuyoGym.ACTION_SIZE)
            next_state, reward, done = env.step(action)
            experience = (state, action, reward, next_state)
            memory.add(experience, gamma, main_qn, target_qn)
            state = next_state

    print('start learn')

    # 学習開始
    for episode in range(num_episodes):   
        env.reset()
        # 最初の一回は適当に行動する
        state, _, done, = env.step(np.random.randint(0, PuyoGym.ACTION_SIZE))

        # 1試行のループ
        while not done:
            # env.render()
            # a = input()
            # 時刻tでの行動を決定する
            action = actor.get_action(state, episode, main_qn)
            next_state, reward, done = env.step(action)
            experience = (state, action, reward, next_state)
            memory.add(experience, gamma, main_qn, target_qn)
            
            # 状態更新
            state = next_state 
            step += 1

            if step % copy_target_freq == 0:
                target_qn.model.set_weights(main_qn.model.get_weights())
                target_qn.model.save_weights(save_weight_path)

            main_qn.replay(memory, batch_size, gamma, target_qn)

        # 1施行終了時の処理
        ojama = -(env.player.common_info.future_ojama.unfixed_ojama + env.player.common_info.future_ojama.fixed_ojama)
        #total_reward_vec = np.hstack((total_reward_vec[1:], ojama))
        total_reward_vec = np.hstack((total_reward_vec[1:], reward))
        #print(f'{episode} Episode finished after {step} steps and {ojama} ojama mean {total_reward_vec.mean()}')
        print(f'{episode} Episode finished after {step} steps and {reward} reward mean {total_reward_vec.mean()}')
        #tensorboard.write(step, ojama, episode)
        tensorboard.write(reward, ojama, episode)

         # 複数施行の平均報酬で終了を判断
        # if total_reward_vec.mean() >= env.goal * 0.9:
        #     print('Episode %d train agent successfuly!' % episode)
        #     env.set_goal(env.goal + 10)

    target_qn.model.save_weights('weights/latest2')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', default='default')
    parser.add_argument('--load_path', default='')
    args = parser.parse_args()
    if args.id == '':
        args.id = 'default'
    run(args.id, args.load_path)
    # prof = LineProfiler()
    # prof.add_function(run)
    # prof.add_function(PuyoGym.make_teacher_label)
    # prof.runcall(run)
    # prof.print_stats()