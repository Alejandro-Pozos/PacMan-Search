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
    Returns a sequence of moves that solv es tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

class Node:
    """A node in a search tree. Contains a pointer to the parent (the node
    that this is a successor of) and to the actual state for this node. Note
    that if a state is arrived at by two paths, then there are two nodes with
    the same state. Also includes the action that got us to this state, and
    the total path_cost (also known as g) to reach the node. Other functions
    may add an f and h value; see best_first_graph_search and astar_search for
    an explanation of how the f and h values are handled. You will not need to
    subclass this class."""

    def __init__(self, state, parent=None, action=None, path_cost=0):
        """Create a search tree Node, derived from a parent by an action."""
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1

    def __repr__(self):
        return "<Node {}>".format(self.state)

    def __lt__(self, node):
        return self.state < node.state

    def expand(self, problem):
        """List the nodes reachable in one step from this node."""
        return [self.child_node(problem, action)
                for action in problem.actions(self.state)]

    def child_node(self, problem, action):
        """[Figure 3.10]"""
        next_state = problem.result(self.state, action)
        next_node = Node(next_state, self, action, problem.path_cost(self.path_cost, self.state, action, next_state))
        return next_node

    def solution(self):
        """Return the sequence of actions to go from the root to this node."""
        return [node.action for node in self.path()[1:]]

    def path(self):
        """Return a list of nodes forming the path from the root to this node."""
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))

    # We want for a queue of nodes in breadth_first_graph_search or
    # astar_search to have no duplicated states, so we treat nodes
    # with the same state as equal. [Problem: this may not be what you
    # want in other contexts.]

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __hash__(self):
        # We use the hash value of the state
        # stored in the node instead of the node
        # object itself to quickly search a node
        # with the same state in a Hash Table
        return hash(self.state)


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    """
    "*** YOUR CODE HERE ***"

    # Start state
    start_node = problem.getStartState()

    # Check if the start state is already in the winning position
    if problem.isGoalState(start_node):
        return []  # Return an empty list if the start state is the goal state

    # Initialize a frontier using a Stack (LIFO) to store nodes to be explored
    frontier = util.Stack()
    frontier.push((start_node, []))  # Push the start node along with an empty list of actions
    
    # Initialize a list to keep track of visited nodes
    visited_nodes = []

    while not frontier.isEmpty():
        # Get the current node and its associated actions from the frontier
        curr_node, actions = frontier.pop()

        # If the current node has not been visited yet
        if curr_node not in visited_nodes:
            # Mark the current node as visited
            visited_nodes.append(curr_node)

            # Check if the current node is the goal state
            if problem.isGoalState(curr_node):
                return actions  # Return the list of actions if the goal state is reached

            # Explore the successors of the current node
            for next_node, action, cost in problem.getSuccessors(curr_node):
                # Create a new set of actions by appending the current action
                next_action = actions + [action]

                # Add the next node along with its actions to the frontier for further exploration
                frontier.push((next_node, next_action))   


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    
    # Get the initial state from the problem
    start_node = problem.getStartState()
    
    # Check if the start node is the goal state
    if problem.isGoalState(start_node):
        return []  # Return an empty list if the start state is the goal state

    # Initialize a frontier as a Queue (FIFO) to store nodes to be explored
    frontier = util.Queue()
    frontier.push((start_node, []))  # Push the start node along with an empty list of actions

    # Initialize a list to keep track of visited nodes
    visited_nodes = []

    # Continue searching until the frontier is not empty
    while not frontier.isEmpty():
        # Get the current node and its associated actions from the frontier
        curr_node, actions = frontier.pop()

        # If the current node has not been visited yet
        if curr_node not in visited_nodes:
            # Mark the current node as visited
            visited_nodes.append(curr_node)

            # Check if the current node is the goal state
            if problem.isGoalState(curr_node):
                return actions  # Return the list of actions if the goal state is reached

            # Explore the successors of the current node
            for next_node, action, cost in problem.getSuccessors(curr_node):
                # Create a new set of actions by appending the current action
                next_action = actions + [action]

                # Add the next node along with its actions to the frontier for further exploration
                frontier.push((next_node, next_action))


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    # Get the start state from the problem
    start_node = problem.getStartState()

    # Check if the start state is already the goal state
    if problem.isGoalState(start_node):
        return []  # Return an empty list if the start state is the goal state

    # Initialize a list to keep track of visited nodes
    visited_nodes = []

    # Initialize a priority queue to explore nodes based on least total cost
    priority_queue = util.PriorityQueue()
    priority_queue.push((start_node, [], 0), 0)  # Push the start node with zero cost initially

    while not priority_queue.isEmpty():
        # Get the current node, its associated actions, and the previous cost from the priority queue
        current_node, actions, prev_cost = priority_queue.pop()

        # If the current node has not been visited yet
        if current_node not in visited_nodes:
            # Mark the current node as visited
            visited_nodes.append(current_node)

            # Check if the current node is the goal state
            if problem.isGoalState(current_node):
                return actions  # Return the list of actions if the goal state is reached

            # Explore the successors of the current node
            for next_node, action, cost in problem.getSuccessors(current_node):
                # Create new actions by appending the current action
                new_actions = actions + [action]
                # Calculate the new total cost
                new_priority = prev_cost + cost

                # Add the next node, its actions, and the new total cost to the priority queue
                priority_queue.push((next_node, new_actions, new_priority), new_priority)

    
def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    # Get the start state from the problem
    start_node = problem.getStartState()

    # Check if the start state is already the goal state
    if problem.isGoalState(start_node):
        return []  # Return an empty list if the start state is the goal state

    # Initialize a list to keep track of visited nodes
    visited_nodes = []

    # Initialize a priority queue to explore nodes based on combined cost and heuristic
    priority_queue = util.PriorityQueue()
    priority_queue.push((start_node, [], 0), 0)  # Push the start node with zero cost initially

    while not priority_queue.isEmpty():
        # Get the current node, its associated actions, and the previous cost from the priority queue
        current_node, actions, prev_cost = priority_queue.pop()

        # If the current node has not been visited yet
        if current_node not in visited_nodes:
            # Mark the current node as visited
            visited_nodes.append(current_node)

            # Check if the current node is the goal state
            if problem.isGoalState(current_node):
                return actions  # Return the list of actions if the goal state is reached

            # Explore the successors of the current node
            for next_node, action, cost in problem.getSuccessors(current_node):
                # Create new actions by appending the current action
                new_actions = actions + [action]
                # Calculate the new cost to reach the next node
                new_cost_to_node = prev_cost + cost
                # Calculate the heuristic cost from the next node to the goal
                heuristic_cost = new_cost_to_node + heuristic(next_node, problem)

                # Calculate the combined cost and heuristic and add it to the priority queue
                priority_queue.push((next_node, new_actions, new_cost_to_node), heuristic_cost)




# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
