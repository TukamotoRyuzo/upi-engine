from keras.layers import Activation, Conv2D, Dense, Input, Flatten, MaxPooling2D, concatenate
from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.utils import plot_model
import upi
import numpy as np
from collections import deque
from line_profiler import LineProfiler
from datetime import datetime
import tensorflow as tf

class QNetwork():
    """
    指し手22種類の中から最善手を選ぶためのニューラルネットワーク。
    時刻tでの状態sにおける行動aによって、この先どの程度の報酬Rがトータルでもらえるのかを返す。
    """
    action_size = 22
    FIELD_SHAPE = (upi.Field.X_MAX, upi.Field.Y_MAX, 5)
    TUMO_SHAPE = (2, 5)
    
    @staticmethod
    def get_batch_field_shape(batch_size):
        return (batch_size, QNetwork.FIELD_SHAPE[0], QNetwork.FIELD_SHAPE[1], QNetwork.FIELD_SHAPE[2])
    
    @staticmethod
    def get_batch_tumo_shape(batch_size):
        return (batch_size, QNetwork.TUMO_SHAPE[0], QNetwork.TUMO_SHAPE[1])

    def __init__(self, learning_rate=0.0001):        
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
        output = Dense(self.action_size, activation='linear', name='output')(all)
        self.model = Model(inputs=[field, tumo_curr, tumo_next], outputs=output)
        self.model.compile(optimizer=Adam(lr=learning_rate), loss='mse')        
        
    def make_teacher_label(self, batch_size, gamma, target_qn, mini_batch):
        field = np.zeros(self.get_batch_field_shape(batch_size))
        tumo_curr = np.zeros(self.get_batch_tumo_shape(batch_size))
        tumo_next = np.zeros(self.get_batch_tumo_shape(batch_size))
        targets = np.zeros((batch_size, QNetwork.action_size))
        for i, (state_b, action_b, reward_b, next_state_b) in enumerate(mini_batch):
            field[i] = state_b[0]
            tumo_curr[i] = state_b[1]
            tumo_next[i] = state_b[2]
            target = reward_b
            if not TokotonEnvironment.state_is_zero(next_state_b):
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
        self.model.fit(inputs, targets, epochs=1, verbose=0)  # epochsは訓練データの反復回数、verbose=0は表示なしの設定


class TokotonEnvironment:
    """
    OpenAIGym形式の環境を作るためのクラス。
    """

    def __init__(self):
        self.player = upi.UpiPlayer()

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

    @staticmethod
    def state_is_zero(state):
        zero = np.zeros(QNetwork.TUMO_SHAPE)
        return np.array_equal(state[1], zero[np.newaxis, :, :])

    @staticmethod
    def get_zero_state():
        state_field = np.zeros(QNetwork.FIELD_SHAPE)
        state_curr_tumo = np.zeros(QNetwork.TUMO_SHAPE)
        state_next_tumo = np.zeros(QNetwork.TUMO_SHAPE)
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
        return ojama >= 500

    def get_reward(self, done):
        if not done:
            return 0        
        if not self.success():
            return -1
        return 1

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
            state = self.get_zero_state()
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
        next_action = np.argmax(main_qn.model.predict(next_state)[0])
        target = reward + gamma * target_qn.model.predict(next_state)[0][next_action]
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
        generatedrand_list = np.random.uniform(0, sum_absolute_td_error,batch_size)
        generatedrand_list = np.sort(generatedrand_list)

        # [※p2]作成した乱数で串刺しにして、バッチを作成する
        idx = 0
        tmp_sum_absolute_td_error = 0
        mini_batch = deque(maxlen=batch_size)
        for randnum in generatedrand_list:
            while tmp_sum_absolute_td_error < randnum:
                tmp_sum_absolute_td_error += abs(self.td_error[idx]) + 0.0001
                idx += 1
            mini_batch.append(self.buffer[idx - 1]) #idx - 1では?
        return mini_batch

class Actor:
    def get_action(self, state, episode, main_qn):
        epsilon = 0.001 + 0.9 / (1.0 + episode)
        #epsilon = 0
        if epsilon <= np.random.uniform(0, 1):
            ret = main_qn.model.predict(state)[0]
            action = np.argmax(ret)
        else:
            action = np.random.choice(np.arange(main_qn.action_size))
        return action


def run():    
    num_episodes = 200000  # 総試行回数
    max_number_of_steps = 100  # 1試行のstep数
    gamma = 0.8 # 割引係数
    memory_size = 1000
    batch_size = 16
    copy_target_freq = 10
    learning_rate = 0.0001
    goal_average_reward = max_number_of_steps * 0.9 # この報酬を超えると学習終了
    num_consecutive_iterations = 10  # 学習完了評価の平均計算を行う試行回数
    total_reward_vec = np.zeros(num_consecutive_iterations)  # 各試行の報酬を格納

    # 環境作成
    env = TokotonEnvironment()        
    main_qn = QNetwork(learning_rate)
    target_qn = QNetwork(learning_rate)
    #main_qn.model.load_weights('weights/latest')
    #target_qn.model.load_weights('weights/latest')
    memory = Memory(max_size=memory_size)
    actor = Actor()

    for episode in range(num_episodes):
        env.reset()
        # 最初の一回は適当に行動する
        state, _, _, = env.step(np.random.randint(0, QNetwork.action_size))
        episode_reward = 0

        if episode % copy_target_freq == 0:
            target_qn.model.set_weights(main_qn.model.get_weights())
            target_qn.model.save_weights('weights/latest')

        # 1試行のループ
        for step in range(max_number_of_steps + 1):
            # env.render()
            # import pdb
            # pdb.set_trace()
            # 時刻tでの行動を決定する
            action = actor.get_action(state, episode, main_qn)
            next_state, reward, done = env.step(action)
            experience = (state, action, reward, next_state)
            memory.add(experience)
            memory.add_td_error(experience, gamma, main_qn, target_qn)

            # 状態更新
            state = next_state 
            episode_reward += 1
            
            if memory.len() > batch_size:
                #main_qn.replay(memory.sample(batch_size), batch_size, gamma, target_qn)
                if total_reward_vec.mean() < max_number_of_steps * 0.01:
                    main_qn.replay(memory.sample(batch_size), batch_size, gamma, target_qn)
                else:
                    main_qn.replay(memory.sample_by_td_error_priority(batch_size), batch_size, gamma, target_qn)

            # 1施行終了時の処理
            if done:
                memory.update_td_error(gamma, main_qn, target_qn)
                total_reward_vec = np.hstack((total_reward_vec[1:], episode_reward))  # 報酬を記録
                #print('%d Episode finished after %d time steps / mean %f' % (episode, step, total_reward_vec.mean()))
                print('%d Episode finished after %d steps and %d ojama / mean %f' % (episode, step,
                -(env.player.common_info.future_ojama.unfixed_ojama + env.player.common_info.future_ojama.fixed_ojama), total_reward_vec.mean()))
                break
    
        # 複数施行の平均報酬で終了を判断
        if total_reward_vec.mean() >= goal_average_reward:
            print('Episode %d train agent successfuly!' % episode)
            break

    target_qn.model.save_weights('weights/latest')

if __name__ == "__main__":
    run()
    # prof = LineProfiler()
    # prof.add_function(run)
    # prof.add_function(QNetwork.make_teacher_label)
    # prof.runcall(run)
    # prof.print_stats()