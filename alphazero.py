from abc import ABC, abstractmethod

class AlphaZeroFramework:
    """
    AlphaZeroアルゴリズムで学習を進めるためのフレームワーク

    Attributes
    ----------
    env: Gym
        対戦型学習環境。
    new_network: NetworkBase
        ニューラルネットワーク。
    actor: Actor
        行動決定を行うクラス。
    memroy: Memory
        経験をためておくクラス。
    """
    def __init__(self, env, network, actor, memory):
        self.env = env
        self.new_network = network
        self.old_network = self.new_network
        self.actor = actor
        self.memory = memory

    def run(self, num_loop, num_episodes, batch_size, epoch, num_battle, update_threshold):
        """
        AlphaZeroアルゴリズムで学習開始。
        """
        for i in range(num_loop):
            self._selfplay(num_episodes)
            self._learn(batch_size, epoch)
            self._evaluate(num_battle, update_threshold)

    def _selfplay(self, num_episodes):
        """
        自己対戦フェーズ。

        Parameters
        ----------
        num_episodes: int
            自己対戦局数。
        """
        for episode in range(num_episodes):   
            self.env.reset()
            state = self.env.get_state()
            done = False
            while not done:
                action = self.actor.get_action(self.env, self.new_network)
                state, reward, done = self.env.step(action)
                experience = (state, action, reward)
                self.memory.add(experience, )

    def _learn(self, batch_size, epoch):
        """
        学習フェーズ。
        """
        self.new_network.fit(batch_size, epoch, memory)
    
    def _evaluate(self, num_battle, update_threshold):
        """
        評価フェーズ。
        """
        win = 0
        lose = 0        
        for num in range(num_battle):
            env.reset()
            state = env.get_state()
            done = False
            while not done:
                action = actor.get_action(env, self.new_network)
                state, reward, done = env.step(action)
            if env.win():
                win += 1
            else:
                lose += 1
        if win / (win + lose) >= update_threshold:
            self.old_network.update(self.new_network)
        else:
            self.new_network.update(self.new_network)

class GymBase(ABC):    
    @abstractmethod
    def get_state(self):
        pass

    @abstractmethod
    def step(self, action, self_network, enemy_network):
        pass

class NetworkBase(ABC):
    @abstractmethod
    def __init__(self, learning_rate): 
        pass

    @abstractmethod
    def fit(self, batch_size, epoch, memory):
        pass

    @abstractmethod
    def update(self, target):
        pass

class ActorBase(ABC):
    @abstractmethod
    def __init__(self): 
        pass
        
    @abstractmethod
    def get_action(self, env, network):
        pass