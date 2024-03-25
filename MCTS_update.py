from captureAgents import CaptureAgent
from capture import GameState
import random, time, util
from game import Directions
import game
import numpy as np
from util import nearestPoint
import copy

import math
import random
import sys
#python capture.py -r baselineTeam -b myTeamTry

class Node:
    def __init__(self, gamestate, agent, action, state = None, parent=None, root=False):
        # by default, nodes are initialised as leaves and as non-terminal states
        self.leaf = True
        self.is_terminal = False
        self.successor_is_terminal = False
        self.root = root

        self.gamestate = gamestate.deepCopy()
        self.parent = parent
        self.action = action  # action that led to this state
        self.children = []
        self.agent = agent
        self.index = agent.index
        if state is None:
            self.state = self.gamestate.getAgentPosition(self.index)
        else:
            self.state = state
        self.legalActions = [act for act in gamestate.getLegalActions(agent.index) if act != 'Stop']   # legal actions from this state
        self.unexploredActions = copy.deepcopy(self.legalActions)
        self.visits = 1
        self.total_score = 0
        self.depth = 0 if parent is None else parent.depth + 1
        self.ucb = 0
        
    def is_leaf(self):
        return len(self.children) == 0
    
    def is_fully_expanded(self):
        return len(self.children) == len(self.legalActions)
    
    def add_child(self, child):
        self.children.append(child)
    
    def select_child(self):
        #exploration_factor = 1.4  # Adjust this parameter as needed
        exploration_factor = 3.4
        best_child = None
        best_score = float('inf')
        
        for child in self.children:
            # UCB formula
            exploitation = child.total_score / child.visits if child.visits != 0 else 0  
            exploration = math.sqrt(math.log(self.visits) / child.visits) if child.visits != 0 else float('-inf')  
            score = exploitation + exploration * exploration_factor  
            self.ucb = score
            
            if score < best_score:  # Lower score is better because we are trying to minimize the score
                best_score = score
                best_child = child

        
        return best_child
    

    def get_best_action(self): 
        # Select the child with the lowest UCB score and return the action that led to that child
        best_child = min(self.children, key=lambda x: x.ucb)
        return best_child.action
    
        
    def get_Rewards(self,action=None):
        # Get the reward for the action taken from the current state
        our_location = self.agent.find_self(self.gamestate)
        successor = self.agent.difference_after_movement(our_location, action)
        food_list = self.find_food(self.gamestate)
        distance = self.agent.getMazeDistance(successor,(30,14))

        # If the successor is the initial position of the agent
        if successor == self.gamestate.getInitialAgentPosition(self.index):
            reward = 10
            return reward
        
        # If we are Pacman
        if self.agent.is_over_half(successor, self.gamestate):  
            # food_list = self.find_food(self.gamestate)
            if self.reached_power_capsule(successor):
                reward = -500  # Reward for reaching power capsule on the enemy field
                return reward
            # If Pacman collected food and returned to its original side, gain points
            food_collected = self.agent.getFoodEaten(self.gamestate)     # Get the number of food pellets eaten by Pacman
            if food_collected > 1:     # If Pacman has eaten at least 4 food pellets
                reward = distance * 10      # We want come back to our field minimising the distance between the agent and the initial position
                return reward
            # If we can catch the food
            if successor in food_list:
                reward = -100 - distance
                return reward
            return - 100 * food_collected - distance # we want to collect food
        
        # If we are ghost
        else:  
            opponents = self.find_opponents(self.gamestate)
            food_list = self.find_food(self.gamestate)
            closest_opponent, closest_food = self.closest_opponent_to_food(food_list, self.find_opponents(self.gamestate))
            # If we can catch the opponent
            if successor in opponents:   
                reward = -5 - distance
                self.successor_is_terminal = True 
                return reward
            # If we are moving towards the opponent
            if self.agent.getMazeDistance(successor, closest_opponent) < self.agent.getMazeDistance(our_location, closest_opponent):
                reward = -3 - distance 
                return reward
            # If we can catch the food
            if successor in food_list:
                reward = -1 - distance 
                return reward
        reward = - distance  # We want to move towards the other side of the board
        return reward
        
    def find_food(self, gameState):
        """
        Finds the location of all our food pallets in the game
        @param gameState: the current game state
        @return: a list of all the food pallets in the game
        """
        food_locations = []

        food = self.agent.getFood(gameState)
        x = 0
        for row in food:
            y = 0
            for cel in row:
                if cel:
                    food_locations.append((x, y))
                y += 1
            x += 1
        
        return food_locations
    
    def reached_power_capsule(self, position):
        """
        Checks if the Pacman agent has reached a power capsule located on the enemy field.
        @param position: The position of the Pacman agent.
        @return: True if the Pacman agent has reached a power capsule on the enemy field, False otherwise.
        """
        capsules = self.agent.getCapsules(self.gamestate)
        for capsule in capsules:
            if self.agent.is_over_half(capsule, self.gamestate) and position == capsule:
                return True
        return False
    
    def find_opponents(self, gameState):
        """
        Finds the location of all the opponents in the game
        @param gameState: the current game state
        @return: a list of all the opponents in the game
        """
        opponents = []
        opponent_index = self.agent.getOpponents(gameState)
        for index in opponent_index:
            opponents.append(gameState.getAgentPosition(index))

        return opponents
    
    def closest_opponent_to_food(self, food_locations, opponents_location):
        min_distance = float('inf')
        closest_opponent = None
        closest_food = None
        for food_loc in food_locations:
            for opp_loc in opponents_location:
                dist = self.agent.getMazeDistance(food_loc, opp_loc)
                if dist < min_distance:
                    min_distance = dist
                    closest_opponent = opp_loc
                    closest_food = food_loc
        return closest_opponent, closest_food
    
    def closest_food_distance(self, our_location, food_locations):
        min_distance = float('inf')
        for food_loc in food_locations:
            dist = self.agent.getMazeDistance(our_location, food_loc)
            if dist < min_distance:
                min_distance = dist
        return min_distance

    def closest_opponent_distance(self, our_location, opponents_location):
        min_distance = float('inf')
        for opp_loc in opponents_location:
            dist = self.agent.getMazeDistance(our_location, opp_loc)
            if dist < min_distance:
                min_distance = dist
        return min_distance
    


class MCTSAgent(CaptureAgent):

    def chooseAction(self, gameState):

        num_iterations = 10  # Adjust this parameter as needed
        root = Node(gameState, agent=self, action = None, state = None, parent=None, root = True)
        
        for _ in range(num_iterations):

            print("iteration: {}".format(_))
            selected_node = self.select(root)
            expanded_node = self.expand(selected_node)
            simulation_result = self.simulate(expanded_node)
            self.backpropagate(expanded_node, simulation_result)
        
        best_action = root.get_best_action()
        print("best action: {} for agent: {}".format(best_action, self.index))
        # sys.exit()
        return best_action
    
    def select(self, node):
        node_selected = node
        while not node_selected.is_leaf():
            if not node_selected.is_fully_expanded():
                return node_selected
            else:
                node_selected = node_selected.select_child()
        return node_selected
    
    def expand(self, node):
        if node.is_terminal:
            print("node is terminal")
            return node
        actions = node.unexploredActions
        if actions == []:
            return node
        random_action = random.choice(actions)
        node.unexploredActions.remove(random_action)
        our_location = self.find_self(node.gamestate)
        new_state = self.difference_after_movement(our_location, random_action)
        new_gamestate = node.gamestate.generateSuccessor(self.index, random_action)
        child_node = Node(gamestate = new_gamestate, state = new_state, agent = self, action = random_action, parent=node)
        node.add_child(child_node)
        return child_node
    

    def simulate(self, node):
        current_node = node
        total_reward = 0

        while not current_node.is_terminal and current_node.depth < 5:

            # We are not considering the enemy agent's actions in the simulation
            actions = current_node.legalActions[:]
            our_location = self.find_self(current_node.gamestate)  # Get the location of our agent

            # Remove the action that leads towards the initial position of the agent to optimise the path
            for action in actions: 
                successor = self.difference_after_movement(our_location, action)
                distance1 = self.getMazeDistance(our_location, (30,14))  # Get the distance between the agent and the initial position
                distance2 = self.getMazeDistance(successor, (30,14))   # Get the distance between the successor and the initial position
                if distance2 < distance1 and len(actions) > 1: # If the successor is closer to the initial position of the agent
                    actions.remove(action)
            random_action = random.choice(actions)
            
            reward = current_node.get_Rewards(action=random_action)
            
            # Update total_reward based on the reward obtained from the current action
            total_reward += reward
            if current_node.successor_is_terminal:
                new_state = self.difference_after_movement(our_location, random_action)
                new_gamestate = current_node.gamestate.generateSuccessor(self.index, random_action)
                current_node = Node(gamestate=new_gamestate, state=new_state, agent=node.agent, action=random_action, parent=current_node)
                current_node.is_terminal = True
                break
            new_state = self.difference_after_movement(our_location, random_action)
            new_gamestate = current_node.gamestate.generateSuccessor(self.index, random_action)
            current_node = Node(gamestate=new_gamestate, state=new_state, agent=node.agent, action=random_action, parent=current_node)
            
        print("total reward: {}".format(total_reward))

        return total_reward


    def backpropagate(self, node, score):
        # Update the total score and the number of visits of the current node and its ancestors
        current_node = node 
        while current_node.root is False:
            current_node.visits += 1
            current_node.total_score += score
            current_node = current_node.parent
            
    def is_over_half(self, position, gameState):
        """
        Checks if the agent is over the half of the board
        @param position: the position of the agent
        @param gameState: the current game state
        @return: True if the agent is over the half of the board, False otherwise
        """
        food = self.getFood(gameState)
        if self.red:
            if position[0] <= len(food[0])-1:
                return False
            else:
                return True
        else:
            if position[0] >= len(food[0]):
                return False
            else:
                return True

    def difference_after_movement(self, pos1, action):
        if action == "North":
            return (pos1[0], pos1[1] + 1)
        elif action == "South":
            return (pos1[0], pos1[1] - 1)
        elif action == "East":
             return (pos1[0] + 1, pos1[1])
        elif action == "West":
            return (pos1[0] - 1, pos1[1])
        else:
            return pos1

    def find_self(self, gameState):
        """
        Finds the location of our agent in the game
        @param gameState: the current game state
        @return: the location of our agent
        """
        return gameState.getAgentPosition(self.index)

    def getFoodEaten(self, gameState):
        """
        Returns the number of food pellets eaten by the agent
        @param gameState: the current game state
        @return: the number of food pellets eaten by the agent
        """
        return gameState.data.agentStates[self.index].numCarrying
       


