from enum import Enum
import numpy as np

class Puyo(Enum):
    """
    ぷよの色を表す定数クラス。
    """
    EMPTY = 0
    RED = 1
    GREEN = 2
    BLUE = 3
    PURPLE = 4
    YELLOW = 5
    OJAMA = 6
    
    @staticmethod
    def to_puyo(character):
        """
        ぷよを表す文字(r|g|b|p|y|o)をPuyoインスタンスに変換する。

        Parameters
        ----------
        character : str
            ぷよを表す文字。
        """
        return Puyo("ergbpyo".find(character))

    def to_str(self):
        """
        Puyoインスタンスをぷよを表す文字(r|g|b|p|y|o)に変換する。
        """
        return "ergbpyo"[int(self.value)]

class Tumo:
    """
    操作対象となっている、上から落ちてくる、ぷよが2つくっついたもの。

    Attributes
    ----------
    pivot : Puyo
        軸ぷよ
    child : Puyo
        子ぷよ
    """
    def __init__(self, c0, c1):
        self.pivot = c0
        self.child = c1


class Rule:
    """
    対戦ルールを表すためのクラス。

    Attributes
    ----------
    falltime : int
        下ボタンを押しっぱなしにしたとき、何フレームで1マス落下するか。
    chaintime : int
        1連鎖につき何フレーム硬直するか。
    settime : int
        ツモ設置時に何フレーム硬直するか。
    nexttime : int
        ツモ設置硬直後、または連鎖終了後、またはお邪魔ぷよが振り終わった後、ネクストが操作可能になるまでに何フレーム硬直するか。
    autodroptime : int
        何も操作しなかったとき、何フレームで1マス落下するか。
    """
    def __init__(self):
        self.chain_time = 60
        self.next_time = 7
        self.set_time = 15
        self.fall_time = 2
        self.autodrop_time = 50

class Move:
    def __init__(self, pivot_sq, child_sq, is_tigiri=False):
        self.pivot_sq = pivot_sq
        self.child_sq = child_sq
        self.is_tigiri = is_tigiri

    def to_upi(self):
        s0 = str(self.pivot_sq[0] + 1)
        s1 = 'abcdefghijklm'[self.pivot_sq[1]]
        s2 = str(self.child_sq[0] + 1)
        s3 = 'abcdefghijklm'[self.child_sq[1]]
        return s0 + s1 + s2 + s3

    @staticmethod
    def none():
        return Move((0, 0), (0, 0))

class Field:
    X_MAX = 6
    Y_MAX = 13    

    # 得点＝Σ{Ａi×（Ｂi＋Ｃi＋Ｄi）}…①
    # iは連鎖数　連鎖ごとに計算が行われる
    # 変数Ａ～Ｄは以下で表される　
    # Ａ＝　消したぷよの数　×１０
    # Ｂ＝　連鎖ボーナス　
    # Ｃ＝  コネクト数　
    # Ｄ＝　同時消し色数ボーナス　
    # 連鎖ボーナス
    CHAIN_BONUS = (0, 8, 16, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512)

    # 連結ボーナス
    CONNECT_BONUS = (0, 2, 3, 4, 5, 6, 7, 10)

    # 同時消し色数ボーナス
    COLOR_BONUS = (0, 3, 6, 12, 24)

    def __init__(self):
        self.field = np.full((self.X_MAX, self.Y_MAX), Puyo.EMPTY)
    
    def init_from_pfen(self, pfen):
        """
        pfen文字列からぷよ配列を初期化する。

        Parameters
        ----------
        pfen : str
            pfen文字列の盤面部分のみ
        """
        x = 0
        y = 0
        self.__init__()
        for p in pfen:
            if p == "/":
                x += 1
                y = 0
            else:
                self.set_puyo(x, y, Puyo.to_puyo(p))
                y += 1

    def set_puyo(self, x, y, col):
        self.field[x, y] = col
    
    def get_puyo(self, x, y):
        return self.field[x, y]

    @staticmethod
    def is_in_field(x, y):
        return x >= 0 and x < Field.X_MAX and y >= 0 and y < Field.Y_MAX - 1

    def pretty_print(self):
        """
        Fieldインスタンスを色付きで見やすく標準出力に表示する。
        """
        color = ('\033[30m', '\033[31m', '\033[32m', '\033[34m', '\033[35m', '\033[33m', '\033[37m')
        END = '\033[0m'        
        pretty_string = self.pretty()
        for p in pretty_string:
            id = "ergbpyo".find(p)
            if id >= 0:
                print(color[id] + p + END, end='')
            else:
                print(p, end='')                
        print('')
        
    def pretty(self):
        result = ''
        for y in reversed(range(self.Y_MAX)):
            for x in range(self.X_MAX):
                result += self.get_puyo(x, y).to_str()
            result += '\r\n'
            if y == 12:
                result += '------\r\n'
        return result[:-2]

    def is_empty(self):
        return np.any(self.field) == Puyo.EMPTY

    def count_connection(self, puyo, x, y, searched):
        if not self.is_in_field(x, y) or searched[x, y] or self.get_puyo(x, y) != puyo:
            return 0     
        searched[x, y] = True
        return (self.count_connection(puyo, x - 1, y, searched) +
                self.count_connection(puyo, x + 1, y, searched) +
                self.count_connection(puyo, x, y - 1, searched) +
                self.count_connection(puyo, x, y + 1, searched) + 1)

    def _delete_puyo_impl(self, chain):
        searched_pos = np.zeros((self.X_MAX, self.Y_MAX), dtype=np.bool)
        delete_pos = np.zeros((self.X_MAX, self.Y_MAX), dtype=np.bool)
        colors = {}
        score = 0
        for x in range(self.X_MAX):
            for y in range(self.Y_MAX - 1):
                puyo = self.get_puyo(x, y)
                if puyo == Puyo.EMPTY:
                    break
                elif not searched_pos[x, y]:
                    searching_pos = np.zeros((self.X_MAX, self.Y_MAX), dtype=np.bool)
                    count = self.count_connection(puyo, x, y, searching_pos)
                    searched_pos |= searching_pos
                    if count >= 4:                        
                        delete_pos |= searching_pos
                        colors[puyo] = 1
                        score += self.CONNECT_BONUS[(count - 4) % 8]
        if len(colors) > 0:
            score += self.CHAIN_BONUS[chain] + self.COLOR_BONUS[len(colors) - 1]
            score = np.count_nonzero(delete_pos) * max(score, 1) * 10
            self.delete_impl(delete_pos)
            self.slide()
        return score

    def delete_impl(self, delete_pos):
        pos = np.where(delete_pos)
        for x, y in zip(pos[0], pos[1]):
            self.set_puyo(x, y, Puyo.EMPTY)
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                if self.is_in_field(x + dx, y + dy) and self.get_puyo(x + dx, y + dy) == Puyo.OJAMA:
                    self.set_puyo(x + dx, y + dy, Puyo.EMPTY)

    def slide(self):
        for x in range(self.X_MAX):
            for y in range(self.Y_MAX):
                if self.get_puyo(x, y) == Puyo.EMPTY:
                    top_y = y
                    while y < self.Y_MAX and self.get_puyo(x, y) == Puyo.EMPTY:
                        y += 1
                    if y >= self.Y_MAX:
                        break                    
                    self.set_puyo(x, top_y, self.get_puyo(x, y))
                    self.set_puyo(x, y, Puyo.EMPTY)

    def delete_puyo(self):
        chain = 0
        score_sum = 0
        while True:
            score = self._delete_puyo_impl(chain)
            if score == 0:
                break
            else:
                chain += 1
                score_sum += score
        return (chain, score_sum)

    def is_death(self):
        return self.get_puyo(2, 11) != Puyo.EMPTY

    def floors(self):
        floor_y = [self.Y_MAX] * 6
        for x in range(self.X_MAX):
            for y in range(self.Y_MAX):
                if self.get_puyo(x, y) == Puyo.EMPTY:
                    floor_y[x] = y
                    break
        return floor_y

class Position:
    """
    Fieldインスタンス、ツモ、スコア、お邪魔を管理するクラス。

    Attributes
    ----------
    field : Field
        Fieldインスタンス。
    tumo_index : int
        ツモ番号。配ツモ自体は外部から引数で与えられる。
    ojama_index : int
        お邪魔ぷよ乱数。
    fall_bonus : int
        落下ボーナス。
    all_clear_flag : bool
        全消しフラグ。
    """

    def __init__(self):
        self.field = Field()
        self.tumo_index = 0
        self.ojama_index = 0
        self.fall_bonus = 0
        self.all_clear_flag = False
        self.rule = Rule()
    
    def fall_ojama(self, positions_common):
        floors = self.field.floors()
        ojama = min(30, positions_common.future_ojama.fixed_ojama)
        while ojama >= Field.X_MAX:
            for x in range(Field.X_MAX):                
                if floors[x] < Field.Y_MAX:
                    self.field.set_puyo(x, floors[x], Puyo.OJAMA)
                    floors[x] += 1
                ojama -= 1
                self.ojama_index = self.ojama_index + 1 % 128
        if ojama > 0:
            v = list(range(6))
            for x in range(Field.X_MAX):
                t = positions_common.tumo_pool[self.ojama_index]
                r = (t.pivot.value + t.child.value) % 6
                v[x], v[r] = v[r], v[x]
                self.ojama_index = self.ojama_index + 1 % 128
            for x in range(ojama):
                if floors[v[x]] < Field.Y_MAX:
                    self.field.set_puyo(v[x], floors[v[x]], Puyo.OJAMA)

    def do_move(self, move, positions_common):
        tumo = positions_common.tumo_pool[self.tumo_index]
        rule = positions_common.rule
        self.tumo_index = (self.tumo_index + 1) % 128
        p = move.pivot_sq
        c = move.child_sq
        self.field.set_puyo(p[0], p[1], tumo.pivot)
        self.field.set_puyo(c[0], c[1], tumo.child)
        chain, score = self.field.delete_puyo()
        if chain > 0:
            if self.all_clear_flag:
                score += 70 * 30
                self.all_clear_flag = False
            if self.field.is_empty():
                self.all_clear_flag = True
            ojama = (score + self.fall_bonus) / 70
            self.fall_bonus = (score + self.fall_bonus) % 70

        drop_frame = max(12 - p[1], 12 - c[1]) * rule.fall_time
        frame = drop_frame + max(abs(2 - p[0]), abs(2 - c[0]))
        + rule.set_time * 2 if move.is_tigiri else rule.set_time
        + rule.chain_time * chain
        + rule.next_time
        if positions_common.future_ojama.time_until_fall_ojama <= frame and positions_common.future_ojama.fixed_ojama > 0:
            self.fall_ojama(positions_common)

def generate_moves(pos, tumo_pool):
    floors = pos.field.floors()
    start_x, end_x = get_move_range(floors)
    moves = []
    tumo = tumo_pool[pos.tumo_index]
    for x in range(start_x, end_x):
        y = floors[x]
        y_side = floors[x + 1]
        dest = (x, y)
        dest_up = (x, y + 1)
        dest_side = (x + 1, y_side)
        is_tigiri = (y != y_side)
        moves.append(Move(dest, dest_up, False))
        moves.append(Move(dest, dest_side, is_tigiri))
        if tumo.pivot != tumo.child:
            moves.append(Move(dest_up, dest, False))
            moves.append(Move(dest_side, dest, is_tigiri))                
    dest = (end_x, floors[end_x])
    dest_up = (end_x, floors[end_x] + 1)
    moves.append(Move(dest, dest_up, False))
    if tumo.pivot != tumo.child:
        moves.append(Move(dest, dest_up, False))
    return moves

def get_move_range(floors):
    left = 0
    right = 5
    for x in reversed(range(2)):
        if floors[x] >= 12:
            left = x + 1
            break
    for x in range(3, Field.X_MAX):
        if floors[x] >= 12:
            right = x - 1
            break
    return (left, right)

def search(pos1, pos2, ojama, depth, frame):
    if pos1.field.is_death():
        return Move.none()
    search_impl(pos1, pos2, ojama, depth, frame, -99999, 99999)

class FutureOjama:
    def __init__(self):
        self.fixed_ojama = 0
        self.unfixed_ojama = 0
        self.time_until_fall_ojama = 0

class PositionsCommonInfo:
    def __init__(self):
        self.tumo_pool = [Tumo(Puyo(i % 4 + 1), Puyo((i + 1) % 4 + 1)) for i in range(128)] # 適当に初期化
        self.rule = Rule()
        self.future_ojama = FutureOjama()

class UpiPlayer:
    def __init__(self):
        self._common_info = PositionsCommonInfo()        
        self._position = [Position(), Position()]

    def upi(self):
        engine_name = "sample_engine"
        version = "1.0"
        author = "Ryuzo Tukamoto"    
        print("id name", engine_name + version)
        print("id author", author)
        print("upiok")

    def tumo(self, tumos):
        self._tumo_pool = [Tumo(Puyo.to_puyo(t[0]), Puyo.to_puyo(t[1])) for t in tumos]
  
    def rule(self, rules):
        for i in range(0, len(rules), 2):
            if rules[i] == "falltime":
                self._common_info.rule.fall_time = int(rules[i + 1])
            elif rules[i] == "chaintime":
                self._common_info.rule.chain_time = int(rules[i + 1])
            elif rules[i] == "settime":
                self._common_info.rule.set_time = int(rules[i + 1])
            elif rules[i] == "nexttime":
                self._common_info.rule.next_time = int(rules[i + 1])
            elif rules[i] == "autodroptime":
                self._common_info.rule.autodrop_time = int(rules[i + 1])

    def isready(self):
        print("readyok")

    def position(self, pfen):
        for i in range(2):
            self._position[i].field.init_from_pfen(pfen[i * 2])            
            self._position[i].tumo_index = int(pfen[i * 2 + 1])
        self._fixed_ojama = int(pfen[4])
        self._unfixed_ojama = int(pfen[5])
        self._time_until_fall_ojama = int(pfen[6])

    def go(self):
        moves = generate_moves(self._position[0], self._tumo_pool)
        print('bestmove', moves[0].to_upi())

    def gameover(self):
        pass

if __name__ == "__main__":
    token = ""
    upi = UpiPlayer()
    while token != "quit":
        cmd = input().split(' ')
        token = cmd[0]
        
        if token == "quit" or token == "stop" or token == "gameover":
            pass

        # UPIエンジンとして認識されるために必要なコマンド
        elif token == "upi":
            upi.upi()

        # 今回のゲームで使うツモ128個
        elif token == "tumo":         
            upi.tumo(cmd[1:]) 

        # ルール
        elif token == "rule": 
            upi.rule(cmd[1:]) 

        # 時間のかかる前処理はここで。
        elif token == "isready": 
            upi.isready() 

        # 思考開始する局面を作る。
        elif token == "position": 
            upi.position(cmd[1:]) 

        # 思考開始の合図。エンジンはこれを受信すると思考を開始。
        elif token == "go": 
            upi.go() 

        # ゲーム終了時に送ってくるコマンド
        elif token == "gameover": 
            upi.gameover()

        # 有効なコマンドではない。
        else:
            print("unknown command: ", cmd)
