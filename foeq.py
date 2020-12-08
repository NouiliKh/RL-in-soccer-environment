import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import cvxpy as cp
from soccer_env import SoccerAgent

class FoeQ(object):
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
        
    def max_min(self, Q, mapped_position_player, mapped_position_opponent, has_ball):
        num_actions = 5
        x = cp.Variable(shape=(num_actions+1, 1), name="x")

        modified_R = np.array(Q[mapped_position_player][mapped_position_opponent][has_ball], dtype="float").T 
        
        modified_R = np.vstack([modified_R, np.eye(num_actions)])
        new_col = [1]*num_actions + [0]* num_actions
        modified_R = np.insert(modified_R, 0, new_col, axis=1) 
    
        A1 = [0] + [1] * num_actions 

        constraints = [cp.matmul(modified_R, x) >= 0, cp.matmul(A1, x) == 1, cp.sum( x[1:]) == 1]
        objective = cp.Minimize(cp.sum(x))
        problem = cp.Problem(objective, constraints)
        solution = problem.solve()
        #print(np.abs(x.value[1:]).reshape((5,) / sum(np.abs(x.value[1:]))[0]))
        return np.abs(x.value[1:]).reshape((5,)) / int(sum(np.abs(x.value[1:]))[0]), np.array(x.value[0])
        
        
        
    def get_action(self, mapped_position_player_A, mapped_position_player_B, A_has_ball, epsilon):
        if np.random.random() < epsilon:  
            action_a =  np.random.randint(5)
            action_b =  np.random.randint(5)
            return action_a, action_b
        else:
            return np.argmax(self.Q_A[mapped_position_player_A][mapped_position_player_B][A_has_ball][np.random.randint(5)][:]), np.argmax(self.Q_B[mapped_position_player_B][mapped_position_player_A][int( not A_has_ball)][np.random.randint(5)][:])
        
        
    def train(self):
        self.Q_A = np.ones((8, 8, 2, 5, 5))
        self.Q_B = np.ones((8, 8, 2, 5, 5))
        
        Pi_A = np.ones((8, 8, 2, 5)) * 1/5
        Pi_B = np.ones((8, 8, 2, 5)) * 1/5
        
        V_A = np.ones((8, 8, 2))
        V_B = np.ones((8, 8, 2))
        

        for episode in range(self.episodes_number):
            self.epsilon = self.epsilon * self.epsilon_decay if self.epsilon > self.epsilon_min else self.epsilon
            self.lr = self.lr * self.lr_decay if self.lr > self.lr_min else self.lr
            position_player_A, position_player_B, A_has_ball = self.env.reset()
            done = False
            mapped_position_player_A = position_player_A[1] *4 + position_player_A[0]
            mapped_position_player_B = position_player_B[1] *4 + position_player_B[0]
            action_player_A, action_player_B = self.get_action(mapped_position_player_A, mapped_position_player_B, A_has_ball, self.epsilon)
            
            current_Q = self.Q_A[6][5][0][2][4]

            while not done:
                new_position_player_A, new_position_player_B, new_A_has_ball, reward, done = self.env.step(action_player_A, action_player_B)
                new_mapped_position_player_A = position_player_A[1] *4 + position_player_A[0]
                new_mapped_position_player_B = position_player_B[1] *4 + position_player_B[0]
                
                new_action_player_A, new_action_player_B = self.get_action(new_mapped_position_player_A, new_mapped_position_player_B, A_has_ball, self.epsilon)
                
                self.Q_A[mapped_position_player_A][mapped_position_player_B][A_has_ball][action_player_B][action_player_A] = self.lr * (reward + (self.gamma* V_A[new_mapped_position_player_A][new_mapped_position_player_B][new_A_has_ball]) - (1-self.lr) * self.Q_A[mapped_position_player_A][mapped_position_player_B][A_has_ball][action_player_B][action_player_A])    
                self.Q_B[mapped_position_player_B][mapped_position_player_A][int(not A_has_ball)][action_player_A][action_player_B] = self.lr * (-reward + (self.gamma* V_B[new_mapped_position_player_B][new_mapped_position_player_A][int(not A_has_ball)]) - self.Q_B[mapped_position_player_B][mapped_position_player_A][int(not A_has_ball)][action_player_A][action_player_B])

                Pi_A[mapped_position_player_A][mapped_position_player_B][A_has_ball], V_A[mapped_position_player_A][mapped_position_player_B][A_has_ball] = self.max_min(self.Q_A, mapped_position_player_A, mapped_position_player_B, A_has_ball)
                Pi_B[mapped_position_player_B][mapped_position_player_A][int(not A_has_ball)], V_B[mapped_position_player_B][mapped_position_player_A][int(not A_has_ball)] =self.max_min(self.Q_B, mapped_position_player_B, mapped_position_player_A, int(not A_has_ball))

    
                action_player_A = new_action_player_A
                action_player_B = new_action_player_B
                
                position_player_A = new_position_player_A
                action_player_B = new_action_player_B
                mapped_position_player_A = new_mapped_position_player_A
                mapped_position_player_B = new_mapped_position_player_B
                
                A_has_ball = new_A_has_ball
                
            new_Q = self.Q_A[6][5][0][2][4]
            self.conv.append(new_Q-current_Q)
        return self.conv

foe_q = FoeQ(1000000).train()
save_stats(foe_q, 'foeq.txt')