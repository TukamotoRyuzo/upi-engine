from keras.layers import Activation, Conv2D, Dense, Input, Flatten, MaxPooling2D, concatenate
from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.utils import plot_model
from keras.callbacks import TensorBoard
from collections import deque
from line_profiler import LineProfiler
from datetime import datetime
from keras import backend as K
import tensorflow as tf
import upi
import numpy as np
import os

# 損失関数にhuber関数を使用します 
# 参考: https://github.com/jaara/AI-blog/blob/master/CartPole-DQN.py
def huberloss(y_true, y_pred):
    err = y_true - y_pred
    cond = K.abs(err) < 1.0
    L2 = 0.5 * K.square(err)
    L1 = (K.abs(err) - 0.5)
    loss = tf.where(cond, L2, L1)  # Keras does not cover where function in tensorflow :-(
    return K.mean(loss)


class QNetwork():
    """
    指し手22種類の中から最善手を選ぶためのニューラルネットワーク。
    時刻tでの状態sにおける行動aによって、この先どの程度の報酬Rがトータルでもらえるのかを返す。
    """
    ACTION_SIZE = 22
    FIELD_SHAPE = (upi.Field.X_MAX, upi.Field.Y_MAX, 5)
    TUMO_SHAPE = (2, 5)
    
    @staticmethod
    def get_batch_field_shape(batch_size):
        return (batch_size, QNetwork.FIELD_SHAPE[0], QNetwork.FIELD_SHAPE[1], QNetwork.FIELD_SHAPE[2])
    
    @staticmethod
    def get_batch_tumo_shape(batch_size):
        return (batch_size, QNetwork.TUMO_SHAPE[0], QNetwork.TUMO_SHAPE[1])

    def __init__(self, learning_rate=0.0001, callbacks=None):        
        # 入力
        # 盤面, ツモ(現在のツモ、次のツモ)
        field = Input(shape=QNetwork.FIELD_SHAPE, name='field')
        tumo_curr = Input(shape=QNetwork.TUMO_SHAPE, name='tumo_curr')
        tumo_next = Input(shape=QNetwork.TUMO_SHAPE, name='tumo_next')

        # ネットワークを定義する
        x = Conv2D(32, 3, activation='relu', padding='same')(field)
        x = Conv2D(32, 3, activation='relu', padding='same')(x)
        #x = MaxPooling2D()(x)
        x = Flatten()(x)
        y = Flatten()(tumo_curr)
        z = Flatten()(tumo_next)
        all = concatenate([x, y, z])
        all = Dense(32, activation='relu')(all) # 適当に1層挟む。
        output = Dense(self.ACTION_SIZE, activation='linear', name='output')(all)
        self.model = Model(inputs=[field, tumo_curr, tumo_next], outputs=output)
        self.model.compile(optimizer=Adam(lr=learning_rate), loss=huberloss)
        #self.model.compile(optimizer=Adam(lr=learning_rate), loss='mse')
        self.callbacks = callbacks
        
    def make_teacher_label(self, batch_size, gamma, target_qn, mini_batch):
        field = np.zeros(self.get_batch_field_shape(batch_size))
        tumo_curr = np.zeros(self.get_batch_tumo_shape(batch_size))
        tumo_next = np.zeros(self.get_batch_tumo_shape(batch_size))
        targets = np.zeros((batch_size, QNetwork.ACTION_SIZE))
        for i, (state_b, action_b, reward_b, next_state_b) in enumerate(mini_batch):
            field[i] = state_b[0]
            tumo_curr[i] = state_b[1]
            tumo_next[i] = state_b[2]
            target = reward_b
            if next_state_b is not None:
                # Double DQN: 今の重みwにより選ばれた行動を、直前の重みw'で評価する
                retmain_qs = self.model.predict(next_state_b)[0]
                next_action = np.argmax(retmain_qs)
                target = reward_b + gamma * target_qn.model.predict(next_state_b)[0][next_action]
            targets[i] = self.model.predict(state_b)[0]
            targets[i][action_b] = target
        return [field, tumo_curr, tumo_next], targets

    # 重みの学習
    def replay(self, mini_batch, batch_size, gamma, target_qn):        
        inputs, targets = self.make_teacher_label(batch_size, gamma, target_qn, mini_batch)        
        self.model.fit(inputs, targets, epochs=1, verbose=0, callbacks=self.callbacks)


class TokotonEnvironment:
    """
    OpenAIGym形式の環境を作るためのクラス。
    """

    def __init__(self):
        self.player = upi.UpiPlayer()
        self.goal = 30

    def set_goal(self, goal):
        self.goal = goal

    def get_state(self):
        """
        ぷよぷよのフィールド、現在のツモ、次のツモを、ニューラルネットワークに入力できる形に変換する。
        """
        # フィールド
        state_field = np.zeros(QNetwork.FIELD_SHAPE)
        for x in range(upi.Field.X_MAX):
            for y in range(upi.Field.Y_MAX):
                puyo = self.player.positions[0].field.get_puyo(x, y)
                if puyo != upi.Puyo.EMPTY and puyo != upi.Puyo.OJAMA:
                    state_field[x, y, puyo.value - 1] = 1
        # ツモ
        state_curr_tumo = np.zeros(QNetwork.TUMO_SHAPE)
        state_next_tumo = np.zeros(QNetwork.TUMO_SHAPE)
        curr_tumo = self.player.common_info.tumo_pool[(self.player.positions[0].tumo_index + 0) % 128]
        next_tumo = self.player.common_info.tumo_pool[(self.player.positions[0].tumo_index + 1) % 128]
        state_curr_tumo[0, curr_tumo.pivot.value - 1] = 1
        state_curr_tumo[1, curr_tumo.child.value - 1] = 1
        state_next_tumo[0, next_tumo.pivot.value - 1] = 1
        state_next_tumo[1, next_tumo.child.value - 1] = 1
        return [state_field[np.newaxis, :, :, :], state_curr_tumo[np.newaxis, :, :], state_next_tumo[np.newaxis, :, :]]

    def action_to_move(self, action):
        """
        actionからmoveを求める。actionは以下の意味とする。
        0~5: 回転数0
        6~11: 回転数2
        12~16: 回転数1
        17~21: 回転数3
        """
        floors = self.player.positions[0].field.floors()
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
        moves = upi.generate_moves(self.player.positions[0], self.player.common_info.tumo_pool)        
        tumo = self.player.common_info.tumo_pool[self.player.positions[0].tumo_index]

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

    def success(self):
        ojama = -(self.player.common_info.future_ojama.unfixed_ojama + self.player.common_info.future_ojama.fixed_ojama)
        return ojama >= self.goal

    def get_reward(self, done):
        if not done:
            return 0

        #ojama = -(self.player.common_info.future_ojama.unfixed_ojama + self.player.common_info.future_ojama.fixed_ojama)
        #reward = ojama / self.goal * 2 - 1.0
        #return reward
        return 1 if self.success() else -1

    def step(self, action):
        """
        actionを基にstateを次の状態に進める。
        """
        # 死んでいる局面でこのメソッドが呼び出されるはずがない。
        assert not self.player.positions[0].field.is_death()
        move = self.action_to_move(action)
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

    def reset(self):
        """
        環境を初期化する。
        """
        self.player.positions[0] = upi.Position()
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


class Memory:
    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)
        self.td_error = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        idx = np.random.choice(np.arange(self.len()), size=batch_size, replace=False)
        return [self.buffer[ii] for ii in idx]

    def len(self):
        return len(self.buffer)

    def add_td_error(self, experience, gamma, main_qn, target_qn):
        self.td_error.append(self.get_td_error(experience, gamma, main_qn, target_qn))

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

    def update_td_error(self, gamma, main_qn, target_qn):
        assert len(self.td_error) == self.len() 
        for i in range(self.len()):
            self.td_error[i] = self.get_td_error(self.buffer[i], gamma, main_qn, target_qn)
    
    # TD誤差の絶対値和を取得
    def get_sum_absolute_td_error(self):
        assert len(self.td_error) == self.len()
        sum_absolute_td_error = 0
        for i in range(self.len()):
            sum_absolute_td_error += abs(self.td_error[i]) + 0.0001  # 最新の状態データを取り出す
        return sum_absolute_td_error

    def sample_by_td_error_priority(self, batch_size):
        # 0からTD誤差の絶対値和までの一様乱数を作成(昇順にしておく)
        sum_absolute_td_error = self.get_sum_absolute_td_error()
        generatedrand_list = np.random.uniform(0, sum_absolute_td_error, batch_size)
        generatedrand_list = np.sort(generatedrand_list)

        # [※p2]作成した乱数で串刺しにして、バッチを作成する
        idx = 0
        tmp_sum_absolute_td_error = 0
        mini_batch = deque(maxlen=batch_size)
        for randnum in generatedrand_list:
            while tmp_sum_absolute_td_error < randnum:
                tmp_sum_absolute_td_error += abs(self.td_error[idx]) + 0.0001
                idx += 1
            mini_batch.append(self.buffer[idx - 1])
        return mini_batch


class Actor:
    def get_action(self, state, episode, main_qn):
        epsilon = 0.001 + 0.9 / (1.0 + episode)
        #epsilon = 0
        if epsilon <= np.random.uniform(0, 1):
            ret = main_qn.model.predict(state)[0]
            action = np.argmax(ret)
        else:
            action = np.random.choice(np.arange(main_qn.ACTION_SIZE))
        return action


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


def run():    
    num_episodes = 200000 # 総試行回数
    max_number_of_steps = 1000  # 1試行のstep数
    gamma = 0.8 # 割引係数
    memory_size = 1000
    batch_size = 16
    copy_target_freq = 1
    learning_rate = 0.0001
    num_consecutive_iterations = 20
    total_reward_vec = np.zeros(num_consecutive_iterations)  # 各試行の報酬を格納

    # tensorboardによる可視化
    log_dir = ".\\logs\\run-{}\\".format(datetime.utcnow().strftime("%Y-%m-%d %H%M%S"))
    tensorboard = TensorBoardLogger(log_dir=log_dir)
    
    # 環境作成
    env = TokotonEnvironment()
    env.set_goal(30)
    main_qn = QNetwork(learning_rate)
    target_qn = QNetwork(learning_rate)
    #main_qn.model.load_weights('weights/latest')
    #target_qn.model.load_weights('weights/latest')
    memory = Memory(max_size=memory_size)
    actor = Actor()
    
    for episode in range(num_episodes):
        env.reset()
        # 最初の一回は適当に行動する
        state, _, _, = env.step(np.random.randint(0, QNetwork.ACTION_SIZE))
        episode_reward = 0

        if episode % copy_target_freq == 0:
            target_qn.model.set_weights(main_qn.model.get_weights())
            target_qn.model.save_weights('weights/latest2')

        # 1試行のループ
        for step in range(max_number_of_steps):
            # env.render()

            # 時刻tでの行動を決定する
            action = actor.get_action(state, episode, main_qn)
            next_state, reward, done = env.step(action)
            experience = (state, action, reward, next_state)
            memory.add(experience)
            #memory.add_td_error(experience, gamma, main_qn, target_qn)

            # 状態更新
            state = next_state 
            episode_reward += 1
            
            if memory.len() > batch_size:
                main_qn.replay(memory.sample(batch_size), batch_size, gamma, target_qn)
                #main_qn.replay(memory.sample_by_td_error_priority(batch_size), batch_size, gamma, target_qn)                

            # 1施行終了時の処理
            if done:
                ojama = -(env.player.common_info.future_ojama.unfixed_ojama + env.player.common_info.future_ojama.fixed_ojama)                
                total_reward_vec = np.hstack((total_reward_vec[1:], ojama))
                #memory.update_td_error(gamma, main_qn, target_qn)                
                print('%d Episode finished after %d steps and %d ojama mean %f' % (episode, step, ojama, total_reward_vec.mean()))
                # tensorboardにloggingする                
                tensorboard.write(step, ojama, episode)
                break

         # 複数施行の平均報酬で終了を判断
        if total_reward_vec.mean() >= env.goal * 0.9:
            print('Episode %d train agent successfuly!' % episode)
            env.set_goal(env.goal + 10)

    target_qn.model.save_weights('weights/latest2')


if __name__ == "__main__":
    run()
    # prof = LineProfiler()
    # prof.add_function(run)
    # prof.add_function(QNetwork.make_teacher_label)
    # prof.runcall(run)
    # prof.print_stats()