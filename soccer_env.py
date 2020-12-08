import pandas as pd
import numpy as np
import random

class SoccerAgent(object):
    def __init__(self):
        self.position_goal_A = 3
        self.position_goal_B = 0
        self.action_mapper = [[0, 0], [0, 1], [1, 0], [-1, 0], [0, -1]]
        
    def reset(self):
        self.done = False
        self.position_player_A = np.array([2, 1])
        self.position_player_B = np.array([1, 1])
        # has_ball =1 => player A has it
        self.has_ball = np.random.choice(1)
        self.score = 0
        return self.position_player_A, self.position_player_B, self.has_ball
    
    
    def check_borders(self, position, action):
        new_position = position + action

        if (new_position[1] == -1) or (new_position[1] == 2) or (new_position[0] == -0) or (new_position[0] == 4):
            return position
        else:
            return new_position
    

    def step(self, action_player_A, action_player_B):
        # action = [0,0] => Stick
        # action = [0,1] => North
        # action = [1,0] => East
        # action = [-1,0] => West
        # action = [0,-1] => South
        action_player_A = self.action_mapper[action_player_A]
        action_player_B = self.action_mapper[action_player_B]

        if np.random.uniform() < 0.5:
            # player A goes first 
            self.position_player_A = self.check_borders(self.position_player_A, action_player_A)
            new_position_player_B = self.check_borders(self.position_player_B, action_player_B)
            if not np.array_equal(self.position_player_A, new_position_player_B):
                self.position_player_B = new_position_player_B
            else:
                self.has_ball = 0
                               
        else:
            self.position_player_B = self.check_borders(self.position_player_B, action_player_B)
            new_position_player_A = self.check_borders(self.position_player_A, action_player_A)
            if not np.array_equal(self.position_player_B, new_position_player_A):
                self.position_player_A = new_position_player_A
            else:
                self.has_ball = 1
    
        # if goal is reached
        if ((self.position_player_A[0] == self.position_goal_A) and (self.has_ball==1)) or ((self.position_player_B[0] == self.position_goal_A) and (self.has_ball==0)):
            self.done = True
            self.score = -100
            return self.position_player_A, self.position_player_B, self.has_ball, self.score, self.done

        elif ((self.position_player_B[0] == self.position_goal_B) and (self.has_ball==1)) or ((self.position_player_A[0] == self.position_goal_B) and (self.has_ball==0)):
            self.done = True
            self.score = 100
            return self.position_player_A, self.position_player_B, self.has_ball, self.score, self.done

        else:
            return self.position_player_A, self.position_player_B, self.has_ball, self.score, self.done
            
