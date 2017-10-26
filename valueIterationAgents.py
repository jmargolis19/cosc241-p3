# valueIterationAgents.py
# -----------------------
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


import mdp, util
import sys

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        #print "DOES THIS HAPPEN? ITERATION %d" % self.iterations
        # Write value iteration code here
        for i in range(0, self.iterations):
          values_copy = self.values.copy()
          delta = 0
          #print "DOES THIS HAPPEN 1"
          for s in self.mdp.getStates():
            if self.mdp.isTerminal(s):
              self.values[s] = self.mdp.getReward(s, None, None)
              continue
            #print "DOES THIS HAPPEN 2"
            
            """
            for a in mdp.getPossibleActions(s):
              print "current state :" + str(s)
              print a
              for s2, p in mdp.getTransitionStatesAndProbs(s, a):
                print s2, p
            """
            #print self.mdp.getReward(s, None, None)
            #print [prob * values_copy[s_next] for (prob, s_next) in self.mdp.getTransitionStatesAndProbs(s, a)] for a in self.mdp.getPossibleActions(s)]
            self.values[s] = self.mdp.getReward(s, None, None) + self.discount * max([ sum([prob * values_copy[s_next] for (s_next, prob) in self.mdp.getTransitionStatesAndProbs(s, a)]) for a in self.mdp.getPossibleActions(s)])
            #delta = max(delta, abs(values_copy[s] - values[s]))

          """
          if delta < self.epsilon * (1 - self.discount) / self.discount:
            return self.values
        
          """
        #print self.values



    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        # REFERENCE: https://docs.google.com/presentation/d/1sLjgsMcDeNcFJbytEtMNbA1rhsVwF1tPbJ_e4zU_JzQ/edit#slide=id.g27271cd14c_0_43
        # CONSTRAINT: Q(s, a) = R(s) + gamma * sum[P(s'|s,a) maxQ(s',a')]
        

        # CHECK HERE
        
        max_so_far = (-1) * float(sys.float_info.max)
        for nextState, p in self.mdp.getTransitionStatesAndProbs(state, action):
            valueNextState = p * self.values[nextState]
            if max_so_far < valueNextState:
              max_so_Far = valueNextState

        return max_so_far

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        if self.mdp.isTerminal(state):
          return None
        else:
          max_so_far = (-1) * float(sys.float_info.max)
          max_action = None
          possibleActions = self.mdp.getPossibleActions(state)
          for action in possibleActions:
            for nextState, p in self.mdp.getTransitionStatesAndProbs(state, action):
              valueNextState = self.getValue(nextState)
              # CHECK: not sure if to multiply p or not
              if max_so_far < p * valueNextState:
                max_so_far = p * valueNextState
                max_action = action

          return max_action 


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
