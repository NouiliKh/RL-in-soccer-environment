import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import cvxpy as cp
from soccer_env import SoccerAgent

class QLearning(object):
    def __init__(self, episodes_number):
        self.env = SoccerAgent()
        self.gamma = 0.9
        self.Q_A = None
        self.Q_B = None
        self.lr = 0.9
        self.lr_min = 0.001
        self.lr_decay = 0.999995
        self.epsilon = 1.0
        self.epsilon_decay = 0.9995
        self.epsilon_min = 0.001
        self.episodes_number = episodes_number
        self.scores = []
        self.conv = []
        
    def get_action(self, mapped_position_player_A, mapped_position_player_B, A_has_ball, epsilon):
        if np.random.random() < epsilon:    
            return np.random.randint(5), np.random.randint(5)
        else:
            return np.argmax(self.Q_A[mapped_position_player_A][mapped_position_player_B][A_has_ball][:]), np.argmax(self.Q_B[mapped_position_player_B][mapped_position_player_A][int( not A_has_ball)][:])
        
        
    def train(self):
        self.Q_A = np.zeros((8, 8, 2, 5))
        self.Q_B = np.zeros((8, 8, 2, 5))
        
        

        for episode in range(self.episodes_number):
            self.epsilon = self.epsilon * self.epsilon_decay if self.epsilon > self.epsilon_min else self.epsilon
            self.lr = self.lr * self.lr_decay if self.lr > self.lr_min else self.lr
            
            position_player_A, position_player_B, A_has_ball = self.env.reset()
            done = False
            mapped_position_player_A = position_player_A[1] *4 + position_player_A[0]
            mapped_position_player_B = position_player_B[1] *4 + position_player_B[0]
            action_player_A, action_player_B = self.get_action(mapped_position_player_A, mapped_position_player_B, A_has_ball, self.epsilon)
            current_Q = self.Q_A[6][5][0][4]

            while not done:

                new_position_player_A, new_position_player_B, new_A_has_ball, reward, done = self.env.step(action_player_A, action_player_B)
                new_mapped_position_player_A = position_player_A[1] *4 + position_player_A[0]
                new_mapped_position_player_B = position_player_B[1] *4 + position_player_B[0]
                
                new_action_player_A, new_action_player_B = self.get_action(new_mapped_position_player_A, new_mapped_position_player_B, A_has_ball, self.epsilon)
                next_Q_A = max(self.Q_A[mapped_position_player_A][mapped_position_player_B][A_has_ball][:])
                next_Q_B = max(self.Q_B[mapped_position_player_B][mapped_position_player_A][int(not A_has_ball)][:])
                
                self.Q_A[mapped_position_player_A][mapped_position_player_B][A_has_ball][action_player_A] += self.lr * (reward + (self.gamma* next_Q_A) - self.Q_A[mapped_position_player_A][mapped_position_player_B][A_has_ball][action_player_A])
                self.Q_B[mapped_position_player_B][mapped_position_player_A][int(not A_has_ball)][action_player_B] += self.lr * (-reward + (self.gamma* next_Q_B) - self.Q_B[mapped_position_player_B][mapped_position_player_A][int(not A_has_ball)][action_player_B])
                
                action_player_A = new_action_player_A
                action_player_B = new_action_player_B
                
                position_player_A = new_position_player_A
                action_player_B = new_action_player_B
                mapped_position_player_A = new_mapped_position_player_A
                mapped_position_player_B = new_mapped_position_player_B
                
                A_has_ball = new_A_has_ball
                
            new_Q = self.Q_A[6][5][0][4]
            self.conv.append(new_Q-current_Q)
        return self.conv

errors = QLearning(1000000).train()
save_stats(errors, 'qlearning.txt')

