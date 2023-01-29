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

        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        states = mdp.getStates()
        startState = mdp.getStartState()
        
        iter_count = 0
        
        while iter_count < iterations:
            current_V_values = util.Counter()
            for state in states:
                if not mdp.isTerminal(state): #if state has no more actions, means no future reward.
                    qValues = [] #all q values from a state into an array
                    for action in mdp.getPossibleActions(state):
                        q = self.computeQValueFromValues(state, action)
                        qValues.append(q)

                    current_V_values[state] = max(qValues) #get V value and add update the old one
            self.values = current_V_values
            iter_count += 1

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
        "*** YOUR CODE HERE ***"
        transitionTuples = self.mdp.getTransitionStatesAndProbs(state, action) #returns state and chance
       
       #function:
       # qValue = Sum T(s,a,s')[R(s,a,s') + rV*(s')]
        qValue = 0.0
       
        for result in transitionTuples:
            futureState, prob = result
            qValue += prob * (self.mdp.getReward(state, action, futureState) + self.discount*self.values[futureState])
            
        return qValue

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        actions = self.mdp.getPossibleActions(state)
        qMax = -10000.0
        bestAction = None
        
        if self.mdp.isTerminal(state):
            return None
        
        for action in actions:
            qValue = self.computeQValueFromValues(state, action)
            if qValue > qMax:
                qMax = qValue
                bestAction = action
        return bestAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
