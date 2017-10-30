# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math,sys

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue 
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        # initialize Q-values for each state-action pair to random values
        self.values = util.Counter() # tuple (s, a) -> v


    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        return self.values[(state,action)]


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        if not self.getLegalActions(state):
          return 0.0
        else:
          maxQValue = (-1) * float(sys.float_info.max)
          maxAction = None
          for a in self.getLegalActions(state):
            currQValue = self.getQValue(state, a)
            if maxQValue < currQValue:
              maxQValue = currQValue
              maxAction = a

          return maxQValue
        

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        
        if not self.getLegalActions(state):
          return None
        else:
          maxQValue = (-1) * float(sys.float_info.max)
          maxAction = None
          for a in self.getLegalActions(state):
            currQValue = self.getQValue(state, a)
            if maxQValue < currQValue:
              maxQValue = currQValue
              maxAction = a

          return maxAction


    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        
        if util.flipCoin(self.epsilon):
          return random.choice(legalActions)
        else:
          return self.computeActionFromQValues(state)

        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        
        # explicit version
        oldQValue = self.getQValue(state, action)
        maxNextAction = None 
        maxNextQValue = (-1.0) * sys.float_info.max
        
        if self.getLegalActions(nextState):
          for a in self.getLegalActions(nextState):
            nextQValue = self.getQValue(nextState, a)
            if maxNextQValue < nextQValue:
              maxNextQValue = nextQValue
              maxNextAction = a
        else: # handling the GAMEOVER state
          maxNextQValue = 0.0

        self.values[(state,action)] = oldQValue + self.alpha * (reward + self.discount * maxNextQValue - oldQValue)

        """
        # shorter version
        if self.getLegalActions(nextState):
          self.values[(state,action)] = oldQValue + self.alpha*(reward + self.discount*max([self.getQValue(nextState, a) for a in self.getLegalActions(nextState)]) - oldQValue)
        else:
          # handling the terminal case separately
          self.values[(state,action)] = oldQValue + self.alpha*(reward - oldQValue)
        """
        
    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """

        total = 0.0

        
        features = self.featExtractor.getFeatures(state, action)
        w = self.getWeights()
        for feature in features:
            total = total + w[feature] * features[feature]
            #if action == 'exit':
              #print "-----------------"
              #print "for (s, a) : " + str((state,action))
              #print total, feature, w[feature], features[feature]
        #print "DISCOUNT RATE? %f" % self.discount
        #print "total = %f" % total
        return total

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        features = self.featExtractor.getFeatures(state, action)
        
        oldQValue = self.getQValue(state, action)
        maxNextAction = None 
        maxNextQValue = (-1.0) * sys.float_info.max
        
        if self.getLegalActions(nextState):
          maxNextQValue = max([self.getQValue(nextState, a) for a in self.getLegalActions(nextState)])
        else: # handling the GAMEOVER state
          maxNextQValue = 0.0


        for feature in features:
            self.weights[feature] = self.weights[feature] + self.alpha * (reward + self.discount * maxNextQValue - oldQValue) * features[feature]
            # BUG RESOLVED HERE:
            # THE PROBLEM WAS THAT I WAS TRYING TO GET MAX QVALUE (WHICH IS BEING UPDATED IN THIS LOOP)
            # IN EACH STEP OF THE LOOP! THE MAX QVALUE MUST BE IDENTIFIED BEFORE THE LOOP.
            
            """
            # for non-terminal cases
            if self.getLegalActions(nextState):

              diff = reward + self.discount*max([self.getQValue(nextState, a) for a in self.getLegalActions(nextState)]) - oldQValue
              self.weights[feature] = self.weights[feature] + self.alpha * diff * features[feature]
            # if the next state is a terminal state 
            else:
              diff = reward - oldQValue
              self.weights[feature] = self.weights[feature] + self.alpha * diff * features[feature]
            """
            
        
        #if action == 'exit':
        #print state, action, self.getQValue(state, action)
        #print self.alpha, reward

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"

            pass
