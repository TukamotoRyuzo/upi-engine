import alphazero
from keras.layers import Activation, Conv2D, Dense, Input, Flatten, MaxPooling2D, concatenate, Lambda
from keras.optimizers import Adam, RMSprop
from keras.models import Sequential, Model
from keras.utils import plot_model
import upi
from collections import deque
import numpy as np
import copy

class PuyoGym(alphazero.GymBase):    
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

    @staticmethod
    def _get_state_impl(position, tumo_pool):
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

    @staticmethod
    def _get_state_battle(position0, position1, common_info):
        state_0 = PuyoGym._get_state_impl(position0, common_info.tumo_pool)
        state_1 = PuyoGym._get_state_impl(position1, common_info.tumo_pool)

        # おじゃまぷよ
        state_ojama = np.zeros(3)
        state_ojama[0] = common_info.future_ojama.fixed_ojama / 1000
        state_ojama[1] = common_info.future_ojama.unfixed_ojama / 1000
        state_ojama[2] = common_info.future_ojama.time_until_fall_ojama / 1000
        return state_0 + state_1 + [state_ojama[np.newaxis]]

    def get_state(self):
        return self._get_state_battle(self.player.positions[0], self.player.positions[1], self.player.common_info)

    def reset(self):
        """
        環境を初期化する。
        """
        self.player.positions[0] = upi.Position()
        self.player.positions[1] = upi.Position()
        self.player.common_info = upi.PositionsCommonInfo()
        self.player.common_info.randomize_tumo()
        return self.get_state()

    def step(self, move):
        """
        actionを基にstateを次の状態に進める。
        """
        # 死んでいる局面でこのメソッドが呼び出されるはずがない。
        assert not self.player.positions[0].field.is_death()
        assert not self.player.positions[1].field.is_death()
        player = 0

        # 相手の番
        if self.player.common_info.time < 0:        
            self.player.common_info.inverse()
            player = 1

        self.player.positions[player].do_move(move, self.player.common_info)

        # 死んだら負け
        if self.player.positions[player].field.is_death():
            return None, -1, True

        # 手番入れ替え
        if player == 1:
            self.player.common_info.inverse()

        state = self.get_state()
        return state, 0, False


class Network(alphazero.NetworkBase):
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

        # 状態価値
        v = Dense(64, activation='relu')(all)
        inputs = [field[0], tumo_curr[0], tumo_next[0], all_clear_flag[0], field[1], tumo_curr[1], tumo_next[1], all_clear_flag[1], ojama]
        outputs = Dense(1, activation='sigmoid')(v)
        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer=RMSprop(lr=learning_rate), loss='mse')
    
    def fit(self, batch_size, epoch, memory):
        inputs, targets = memory.sample()
        self.model.fit(inputs, targets, batch_size=batch_size, epochs=epoch, verbose=0)

    def update(self, target):
        self.model.set_weights(target.model.get_weights())

    def predict(self, state):
        value = self.model.predict(state)[0]
        return value


class Actor(alphazero.ActorBase):
    def __init__(self): 
        self.network = None
        
    def get_action(self, env, network):
        self.network = network
        move = self.search(env.player.positions[0], env.player.positions[1], env.player.common_info, 2)        
        return move

    def search(self, pos1, pos2, positions_common, depth):
        """
        この局面での最善手を探索する。

        Parameters
        ----------
        pos1 : Position
            1Pの局面。
        pos2 : Position
            2Pの局面。
        positions_common : PositionsCommonInfo
            1Pと2Pで共通のデータ。
        depth : int
            探索深さ。

        Returns
        -------
        move : Move
            探索した結果の指し手。
        """
        if pos1.field.is_death():
            return Move.none()
        moves = upi.generate_moves(pos1, positions_common.tumo_pool)
        _, move = self.search_impl(pos1, pos2, positions_common, depth)
        if move.to_upi() == upi.Move.none().to_upi():
            return moves[0]
        return move

    def search_impl(self, pos1, pos2, positions_common, depth):
        """
        evaluateの返り値が最も大きくなる手を探索する。

        Parameters
        ----------
        pos1 : Position
            1Pの局面。
        pos2 : Position
            2Pの局面。
        positions_common : PositionsCommonInfo
            1Pと2Pで共通のデータ。
        depth : int
            探索深さ。

        Returns
        -------
        best_score : int
            最も大きかったevaluateの返り値。
        best_move : Move
            best_scoreを得ることができる指し手。
        """

        if depth == 0:
            return self.evaluate(pos1, pos2, positions_common), upi.Move.none()
        if pos1.field.is_death():
            return -999999, upi.Move.none()

        moves = upi.generate_moves(pos1, positions_common.tumo_pool)
        best_score = -999999
        best_move = upi.Move.none()
        for move in moves:
            pos = copy.deepcopy(pos1)
            com = copy.copy(positions_common)
            com.future_ojama = copy.deepcopy(positions_common.future_ojama)
            pos.do_move(move, com)
            score, _ = self.search_impl(pos, pos2, com, depth - 1)
            if score > best_score:
                best_score = score
                best_move = move
        return best_score, best_move

    def evaluate(self, pos0, pos1, positions_common):
        """
        局面を評価する。単純に、お邪魔ぷよの数で判定する。

        Parameters
        ----------
        pos : Position
            評価する局面。
        positions_common : PositionsCommonInfo
            共通データ。お邪魔ぷよを含む。

        Returns
        -------
        eval : int
            局面のスコア。今のところお邪魔ぷよの数。相手に降らせる数が多いほどハイスコア。
        """
        if pos0.field.is_death():
            return -999999
        if pos1.field.is_death():
            return 999999
        value = self.network.predict(PuyoGym._get_state_battle(pos0, pos1, positions_common))
        return value


class Memory:
    """
    経験再生用メモリ。
    """
    def __init__(self, max_size=5000):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, target):
        self.buffer.append((state, target))

    def sample(self, batch_size):
        idx = np.random.choice(np.arange(self.len()), size=batch_size, replace=False)
        return [(i, self.buffer[ii]) for i, ii in enumerate(idx)]

    def len(self):
        return len(self.buffer)

if __name__ == "__main__":
    env = PuyoGym()    
    network = Network()
    actor = Actor()
    memory = Memory()
    alphazero = alphazero.AlphaZeroFramework(env, network, actor, memory)
    num_loop = 10000
    num_episodes = 25000
    batch_size = 32
    epoch = 8
    num_battle = 300
    update_threshold = 0.55
    alphazero.run(num_loop, num_episodes, batch_size, epoch, num_battle, update_threshold)