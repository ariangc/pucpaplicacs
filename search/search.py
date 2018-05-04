# search.py
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from Queue import *

class SearchProblem:
	"""
	This class outlines the structure of a search problem, but doesn't implement
	any of the methods (in object-oriented terminology: an abstract class).

	You do not need to change anything in this class, ever.
	"""

	def getStartState(self):
		"""
		Returns the start state for the search problem.
		"""
		util.raiseNotDefined()

	def isGoalState(self, state):
		"""
		  state: Search state

		Returns True if and only if the state is a valid goal state.
		"""
		util.raiseNotDefined()

	def getSuccessors(self, state):
		"""
		  state: Search state

		For a given state, this should return a list of triples, (successor,
		action, stepCost), where 'successor' is a successor to the current
		state, 'action' is the action required to get there, and 'stepCost' is
		the incremental cost of expanding to that successor.
		"""
		util.raiseNotDefined()

	def getCostOfActions(self, actions):
		"""
		 actions: A list of actions to take

		This method returns the total cost of a particular sequence of actions.
		The sequence must be composed of legal moves.
		"""
		util.raiseNotDefined()


def tinyMazeSearch(problem):
	"""
	Returns a sequence of moves that solves tinyMaze.  For any other maze, the
	sequence of moves will be incorrect, so only use this for tinyMaze.
	"""
	from game import Directions
	s = Directions.SOUTH
	w = Directions.WEST
	return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
	"""
	Search the deepest nodes in the search tree first.

	Your search algorithm needs to return a list of actions that reaches the
	goal. Make sure to implement a graph search algorithm.

	To get started, you might want to try some of these simple commands to
	understand the search problem that is being passed in:

	print "Start:", problem.getStartState()
	print "Is the start a goal?", problem.isGoalState(problem.getStartState())
	print "Start's successors:", problem.getSuccessors(problem.getStartState())
	"""
	"*** YOUR CODE HERE ***"
	vis, state, answer = [], problem.getStartState(), []
	DFS(problem, state, vis, answer)
	#print([x[0] for x in answer])
	return answer

def DFS(problem, state, vis, answer):
	if(problem.isGoalState(state)):
		return True

	adj = problem.getSuccessors(state)
	for nextState in adj:
		if(nextState[0] in vis): #If the state has been already visited, skip
			continue
		vis.append(nextState[0])
		answer.append(nextState[1])
		if DFS(problem, nextState[0], vis, answer):
			return True
		answer.pop()

	return False


def breadthFirstSearch(problem):
	"""Search the shallowest nodes in the search tree first."""
	"*** YOUR CODE HERE ***"
	q = Queue()
	vis, state, answer, parent = [], problem.getStartState(), [], {}
	#print(state)
	q.put(state)
	vis.append(state)
	currState = None
	parent[state] = None
	while(not q.empty()):
		currState = q.get()
		if problem.isGoalState(currState):
			break
		adj = problem.getSuccessors(currState)
		for nextState,action,cost in adj:
			if nextState not in vis:
				vis.append(nextState)
				parent[nextState] = (currState, action)
				q.put(nextState)

	while parent[currState]:
		answer.append(parent[currState][1])
		currState = parent[currState][0]

	answer = answer[::-1] #Reverse
	#print([x[0] for x in answer])
	return answer

def uniformCostSearch(problem):
	"""Search the node of least total cost first."""
	"*** YOUR CODE HERE ***"
	util.raiseNotDefined()

def nullHeuristic(state, problem=None):
	"""
	A heuristic function estimates the cost from the current state to the nearest
	goal in the provided SearchProblem.  This heuristic is trivial.
	"""
	return 0

def aStarSearch(problem, heuristic=nullHeuristic):
	"""Search the node that has the lowest combined cost and heuristic first."""
	"*** YOUR CODE HERE ***"
	q = PriorityQueue()
	vis, state, answer, parent, dist = [], problem.getStartState(), [], {}, {}
	#print(state)
	q.put([heuristic(state, problem),state])
	vis.append(state)
	dist[state] = 0
	parent[state] = None
	currState = None
	while(not q.empty()):
		currState = q.get()
		currState = currState[1]
		if problem.isGoalState(currState):
			break
		adj = problem.getSuccessors(currState)
		for nextState, action, cost in adj:
			if nextState not in vis:
				dist[nextState] = dist[currState] + cost
				q.put([(heuristic(nextState, problem) + dist[nextState]), nextState])
				parent[nextState] = (currState, action)
				vis.append(nextState)
			else:
				if(dist[nextState] > dist[currState] + cost):
					dist[nextState] = dist[currState] + cost
					q.put([(heuristic(nextState, problem) + dist[nextState]), nextState])
					parent[nextState] = (currState, action)

	while parent[currState]:
		answer.append(parent[currState][1])
		currState = parent[currState][0]

	answer = answer[::-1] #Reverse
	print([x[0] for x in answer])
	return answer


def iterativeDeepeningSearch(problem):
	""" Iterative Deepening Search improved with Binary Search to find the lowest depth"""
	minDepth, maxDepth = 0, int(1e5)
	while minDepth < maxDepth:
		depth = (minDepth + maxDepth) >> 1
		print("Depth: {}".format(depth))
		vis, state, answer = [], problem.getStartState(), []
		found = DLS(problem, state,vis,answer, depth)
		print(depth, found)
		if found:
			maxDepth = depth
		else:
			minDepth = depth + 1

	vis, state, answer = [], problem.getStartState(), []
	found = DLS(problem, state,vis,answer, maxDepth)
	return answer

def DLS(problem, state, vis, answer, depth):
	if problem.isGoalState(state):
		return True

	if depth > 0:
		adj = problem.getSuccessors(state)
		for nextState in adj:
			if(nextState[0] in vis): #If the state has been already visited, skip
				continue
			vis.append(nextState[0])
			answer.append(nextState[1])
			if DLS(problem, nextState[0], vis, answer, depth-1):
				return True
			answer.pop()

	return False

def bidirectionalSearch(problem):
	qI = []
	qG = []
	visI,visG, answer, parent,child =[], [], [], {},{}
	#print(state)
	qI.insert(0,problem.getStartState())
	qG.insert(0,(problem.startingPosition,tuple([1,1,1,1])))
	visI.append(problem.getStartState())
	visG.append((problem.startingPosition,tuple([1,1,1,1])))
	currStateI = None
	currStateG = None
	parent[problem.getStartState()] = None
	#parent[(problem.startingPosition,tuple([1,1,1,1]))] = None
	while((not len(qI)==0) and (not len(qG)==0)):
		currStateI = qI.pop()
		if problem.isGoalState(currStateI) or (currStateI in qG):
			print "waaa "
			break
		adjI = problem.getSuccessors(currStateI)
		for nextStateI,actionI,costI in adjI:
			if nextStateI not in visI:
				visI.append(nextStateI)
				parent[nextStateI] = (currStateI, actionI)
				qI.insert(0,nextStateI)

		currStateG = qG.pop()
		if currStateG==problem.getStartState() or (currStateG in qI):
			print "weee "
			break
		adjG = problem.getPredecessors(currStateG)
		for nextStateG,actionG,costG in adjG:
			if nextStateG not in visG:
				visG.append(nextStateG)
				child[nextStateI] = (currStateG, actionG)
				qG.insert(0,nextStateG)

	print "sali :'v'"

	#stateF=(problem.startingPosition,tuple([1,1,1,1]))
	aux=currStateI
	while parent[currStateI]:
		answer.append(parent[currStateI][1])
		currStateI = parent[currStateI][0]
		print currStateI

	answer = answer[::-1] #Reverse

	while child[aux]:
		answer.append(child[aux][1])
		aux = child[aux][0]


	#print([x[0] for x in answer])
	return answer

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
ids = iterativeDeepeningSearch
bds = bidirectionalSearch
