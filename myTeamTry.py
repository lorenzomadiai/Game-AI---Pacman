# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from captureAgents import CaptureAgent
from capture import GameState
import random, time, util
from game import Directions
import game
from MCTS_MOD import Node
from MCTS_MOD import MCTSAgent
import sys

#################
# Team creation #
#################

# def createTeam(firstIndex, secondIndex, isRed,
#                first = 'DummyAgent', second = 'DummyAgent'):
#   """
#   This function should return a list of two agents that will form the
#   team, initialized using firstIndex and secondIndex as their agent
#   index numbers.  isRed is True if the red team is being created, and
#   will be False if the blue team is being created.

#   As a potentially helpful development aid, this function can take
#   additional string-valued keyword arguments ("first" and "second" are
#   such arguments in the case of this function), which will come from
#   the --redOpts and --blueOpts command-line arguments to capture.py.
#   For the nightly contest, however, your team will be created without
#   any extra arguments, so you should make sure that the default
#   behavior is what you want for the nightly contest.
#   """

#   # The following line is an example only; feel free to change it.
#   return [eval(first)(firstIndex), eval(second)(secondIndex)]

def createTeam(firstIndex, secondIndex, isRed,
               first = 'MCTSPacmanAgent', second = 'MCTSGhostAgent'):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.
    """

    # The following line is an example only; feel free to change it.
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########

# class DummyAgent(CaptureAgent):
#   """
#   A Dummy agent to serve as an example of the necessary agent structure.
#   You should look at baselineTeam.py for more details about how to
#   create an agent as this is the bare minimum.
#   """

#   def registerInitialState(self, gameState):
#     """
#     This method handles the initial setup of the
#     agent to populate useful fields (such as what team
#     we're on).

#     A distanceCalculator instance caches the maze distances
#     between each pair of positions, so your agents can use:
#     self.distancer.getDistance(p1, p2)

#     IMPORTANT: This method may run for at most 15 seconds.
#     """

#     '''
#     Make sure you do not delete the following line. If you would like to
#     use Manhattan distances instead of maze distances in order to save
#     on initialization time, please take a look at
#     CaptureAgent.registerInitialState in captureAgents.py.
#     '''
#     CaptureAgent.registerInitialState(self, gameState)

#     '''
#     Your initialization code goes here, if you need any.
#     '''


#   def chooseAction(self, gameState):
#     """
#     Picks among actions randomly.
#     """
#     actions = gameState.getLegalActions(self.index)

#     '''
#     You should change this in your own agent.
#     '''

#     return random.choice(actions)
  
# import random
# from captureAgents import CaptureAgent

NUM_SIMULATIONS = 100


class MCTSPacmanAgent(MCTSAgent):
    """
    A Pacman agent that uses Monte Carlo Tree Search (MCTS) to make decisions.
    """
    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)

    def chooseAction(self, gameState):
        """
        Choose an action using MCTS.
        """
        our_location = gameState.getAgentPosition(self.index)
        pacman = self.is_over_half(our_location, gameState)
        gameState = GameState.deepCopy(self=gameState)
        best_action = MCTSAgent.chooseAction(self, gameState, pacman)
        return best_action
    
class MCTSGhostAgent(MCTSAgent):
    """
    A Ghost agent that uses Monte Carlo Tree Search (MCTS) to make decisions.
    """
    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)

    def chooseAction(self, gameState):
        """
        Choose an action using MCTS.
        """
        our_location = gameState.getAgentPosition(self.index)
        pacman = self.is_over_half(our_location, gameState)
        gameState = GameState.deepCopy(self=gameState)
        best_action = MCTSAgent.chooseAction(self, gameState, pacman)
        return best_action



        
