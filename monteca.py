class MCTSNode(object):

    def __init__(self, gameState, agent, action, parent, enemy_pos, borderline):
        self.parent = parent
        self.action = action
        self.depth = parent.depth + 1 if parent else 0

        self.child = []
        self.visits = 1
        self.q_value = 0.0

        self.gameState = gameState.deepCopy()
        self.enemy_pos = enemy_pos
        self.legalActions = [act for act in gameState.getLegalActions(agent.index) if act != 'Stop']
        self.unexploredActions = self.legalActions[:]
        self.borderline = borderline

        self.agent = agent
        self.epsilon = 1
        self.rewards = 0

    def node_expanded(self):
        if self.depth >= max_depth:
            return self

        if self.unexploredActions != []:
            action = self.unexploredActions.pop()
            current_game_state = self.gameState.deepCopy()
            next_game_state = current_game_state.generateSuccessor(self.agent.index, action)
            child_node = MCTSNode(next_game_state, self.agent, action, self, self.enemy_pos, self.borderline)
            self.child.append(child_node)
            return child_node

        if util.flipCoin(self.epsilon):
            next_best_node = self.sel_best_child()
        else:
            next_best_node = random.choice(self.child)
        return next_best_node.node_expanded()

    def mcts_search(self):
        timeLimit = 0.99
        start = time.time()
        while (time.time() - start < timeLimit):

            node_selected = self.node_expanded()

            reward = node_selected.cal_reward()

            node_selected.backpropagation(reward)

        return self.sel_best_child().action

    def sel_best_child(self):
        best_score = -np.inf
        best_child = None
        for candidate in self.child:
            score = candidate.q_value / candidate.visits
            if score > best_score:
                best_score = score
                best_child = candidate
        return best_child

    def backpropagation(self, reward):
        self.visits += 1
        self.q_value += reward
        if self.parent is not None:
            self.parent.backpropagation(reward)

    def cal_reward(self):
        current_pos = self.gameState.getAgentPosition(self.agent.index)
        if current_pos == self.gameState.getInitialAgentPosition(self.agent.index):
            return -1000
        value = self.get_feature() * MCTSNode.get_weight_backup(self)
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
