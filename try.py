from captureAgents import CaptureAgent
from capture import GameState
import random, time, util
from game import Directions
import game
import numpy as np

import math

import numpy
import random

# the node class stores a list of available moves
# and the associated play counts and scores for
# each move. 

class Node():
  # by default, nodes are initialised as leaves and as non-terminal states
  def __init__(self, gamestate, agent, action, parent, enemy_pos, borderline):
    self.leaf = True
    self.terminal = False

    self.parent = parent
    self.action = action
    self.depth = parent.depth + 1 if parent else 0

    self.children = []

    self.gameState = GameState.deepCopy()
    self.enemy_pos = enemy_pos
    self.legalActions = [act for act in GameState.getLegalActions(agent.index) if act != 'Stop']
    self.unexploredActions = self.legalActions[:]
    self.borderline = borderline

    self.agent = agent
    self.epsilon = 1
    self.rewards = 0


  # A node is expanded using a list of moves.
  # The node is terminal if there are no moves (game drawn).
  # Otherwise, the scores for the moves are initialised to zero
  # and the counters to a small number to avoid division by zero.
  # One child is created for each move.
  def expand(self):
    moves = self.unexploredActions
    m = len(moves)
    if self.depth >= max_depth:
            return self 
    if m == 0:
      self.terminal = True
    else:
      # S stores the cumulative reward for each of the m moves 
      self.S = numpy.zeros(m)

      # T stores the number of plays for each of the m moves
      self.T = numpy.full(m, 0.001)
      
      current_game_state = self.gameState.deepCopy()
      action = self.unexploredActions.pop()
      next_game_state = current_game_state.generateSuccessor(self.agent.index, action)
      self.children = [Node(next_game_state, action, agent=self.agent, parent = self.parent, enemy_pos = self.enemy_pos, borderline=self.borderline ) for a in range(m)]
      self.leaf = False

  # when the move associated with idx is played
  # and a score obtained, this function should
  # be used to update the counts (T) and scores (S)
  # associated with the idx
  def update(self):
    # Calculate UCB scores for each move
    self.ucb_scores = self.S / self.T + numpy.sqrt(2 * numpy.log(sum(self.T)) / self.T)
    for act in self.legalActions:
      if act != 'Stop':
        idx = self.legalActions.index(act)
        self.S[idx]+=self.ucb_scores[idx]
        self.T[idx]+=1
    
  def cal_reward(self):
        current_pos = self.gameState.getAgentPosition(self.agent.index)
        if current_pos == self.gameState.getInitialAgentPosition(self.agent.index):
            return -1000
        value = self.get_feature() * self.get_weight_backup(self)
        return value
  
  def get_feature(self):
        gameState = self.gameState
        feature = util.Counter()
        current_pos = self.gameState.getAgentPosition(self.agent.index)
        feature['distance'] = min([self.agent.getMazeDistance(current_pos, borderPos) for borderPos in self.borderline])
        return feature
  
  def get_weight(self):
        return {'minDistToFood': -10, 'getFood': 100}
  
  def get_weight_backup(self):
        return {'distance': -1}



      
    
  # THIS FUNCTION NEEDS IMPLEMENTING
  # This function decides which node to search using a bandit algorithm
  # Notes:
  # (1) self.S stores the cumulative returns of each available move at this node
  # (2) self.T stores the number of times each available move has been explored
  # (3) numpy.argmax(v) returns the coordinate the maximises vector v.
  # (4) numpy is quite permissive about operations on vectors. When v and w have the same size, then v/w divides
  # coordinatewise. Similarly, numpy.sqrt(v) is a coordinate-wise operation.
  # (5) You might like experimenting with this function
  def choose(self):

    # Choose the move with the highest UCB score
    return numpy.argmax(self.ucb_scores)


class MCTS():
  def __init__(self, iterations = 5000):
    self.game = GameState.deepCopy()
    self.iterations = iterations

  def act(self):
    move = self.search() 
    return move

  def feed(self, move):
    self.game.make_move(move)

  def search(self):
    print("MCTS searching")

    # create the root note
    node = Node() 

    # repeat a number of iterations of MCTS
    for t in range(self.iterations):
      self.mcts(node)

    # find the index of the most played move at the root node and associated move
    idx = numpy.argmax(node.T)
    move = node.moves[idx]

    # print the average return of this move and return the move
    print("MCTS score: " + str(node.S[idx] / node.T[idx]))
    return move


  def mcts(self, node):
    # if the node is a leaf, then expand it
    if node.leaf:
      node.expand(self.game.moves())
      rollout = True
    else:
      rollout = False

    if node.terminal:
      return 0

    # choose a move 
    idx = node.choose()
    move = node.moves[idx]
    if self.game.make_move(move): # if the move wins, the value is 1
      val = 1
    elif rollout:                 # if we just expanded a node, then get value from rollout
      val = -self.rollout()
    else:                         # otherwise continue traversing the tree
      val = -self.mcts(node.children[idx])
    self.game.unmake_move()

    # update the value of the associated move
    node.update(idx, val)
    return val

  # THIS FUNCTION NEEDS IMPLEMENTING
  # returns the result of a Monte-Carlo rollout from the current game state
  # until the game ends. After the function has returned, the game-state should
  # not be different than it was at the start.
  # The value should be from the perspective of the player currently to move:
  # (1) If the player wins, it should be 1
  # (2) If the player loses, it should be -1
  # (3) If the game was drawn, it should be 0
  # Note:
  # self.game.moves() returns the moves available in the current position
  # self.game.make_move(move) makes move on self.game and returns True if that move won the game
  # self.game.unmake_move() unmakes the last move in self.game
  # Be care that rollout should return the score from the perspective of the player currently to move
  def rollout(self):
    assert(False)

  # You can call this function to test your rollout function
  # Running it should give results of about 0.1 +- 0.05 and -0.03 +- 0.01
  def test_rollout(self):
    assert(len(self.game.history) == 0)
    n = 5000
    res = 0.0
    for t in range(n):
      res+=self.rollout()
      assert(len(self.game.history) == 0)
    print("average result from start with random play: " + str(res / n)) 
    self.game.make_move(0)
    res = 0.0
    for t in range(n):
      res+=self.rollout()
      assert(len(self.game.history) == 1)
    print("average result after first player moves 0: " + str(res / n))















