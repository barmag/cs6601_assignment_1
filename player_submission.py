#!/usr/bin/env python
from isolation import Board, game_as_text
from random import randint

# This file is your main submission that will be graded against. Do not
# add any classes or functions to this file that are not part of the classes
# that we want.


class OpenMoveEvalFn:

	def score(self, game, maximizing_player_turn=True):
		"""Score the current game state

		Evaluation function that outputs a score equal to how many
		moves are open for AI player on the board minus how many moves
	are open for Opponent's player on the board.

	Note:
		1. Be very careful while doing opponent's moves. You might end up
		   reducing your own moves.
		2. Here if you add overlapping moves of both queens, you are considering one available square twice.
		   Consider overlapping square only once. In both cases- myMoves and in OppMoves.
		3. If you think of better evaluation function, do it in CustomEvalFn below.

		Args
			param1 (Board): The board and game state.
			param2 (bool): True if maximizing player is active.

		Returns:
			float: The current state's score. MyMoves-OppMoves.

		"""

	# TODO: finish this function!
		player_moves = len(game.get_legal_moves())
		opponent_moves = len(game.get_opponent_moves())
		return float(player_moves - opponent_moves)



class CustomEvalFn:

	def __init__(self):
		pass

	def score(self, game, maximizing_player_turn=True):
		"""Score the current game state

		Custom evaluation function that acts however you think it should. This
		is not required but highly encouraged if you want to build the best
		AI possible.

		Args
			game (Board): The board and game state.
			maximizing_player_turn (bool): True if maximizing player is active.

		Returns:
			float: The current state's score, based on your own heuristic.

		"""

		# TODO: finish this function!
		raise NotImplementedError


class CustomPlayer:
	# TODO: finish this class!
	"""Player that chooses a move using
	your evaluation function and
	a minimax algorithm
	with alpha-beta pruning.
	You must finish and test this player
	to make sure it properly uses minimax
	and alpha-beta to return a good move."""

	def __init__(self, search_depth, eval_fn=OpenMoveEvalFn()):
		"""Initializes your player.

		if you find yourself with a superior eval function, update the default
		value of `eval_fn` to `CustomEvalFn()`

		Args:
			search_depth (int): The depth to which your agent will search
			eval_fn (function): Utility function used by your agent
		"""
		self.eval_fn = eval_fn
		self.search_depth = search_depth

	def move(self, game, legal_moves, time_left):
		best_move_queen1, best_move_queen2, utility = self.minimax(game, time_left, depth=self.search_depth)
		return best_move_queen1, best_move_queen2
	"""Called to determine one move by your agent

	Note:
		1. Do NOT change the name of this 'move' function. We are going to call
		the this function directly.
		2. Change the name of minimax function to alphabeta function when
		required. Here we are talking about 'minimax' function call,
		NOT 'move' function name.

		Args:
			game (Board): The board and game state.
			legal_moves (dict): Dictionary of legal moves and their outcomes
			time_left (function): Used to determine time left before timeout

		Returns:
			(tuple, tuple): best_move_queen1, best_move_queen2
		"""

	def utility(self, game, maximizing_player):
		"""Can be updated if desired. Not compulsory. """
		return self.eval_fn.score(game)

	def minimax(self, game, time_left, depth, maximizing_player=True):
		"""Implementation of the minimax algorithm

		Args:
			game (Board): A board and game state.
			time_left (function): Used to determine time left before timeout
			depth: Used to track how deep you are in the search tree
			maximizing_player (bool): True if maximizing player is active.

		Returns:
			(tuple,tuple, int): best_move_queen1,best_move_queen2, val
		"""
	# TODO: finish this function!
		raise NotImplementedError
		return best_move_queen1,best_move_queen2, best_val

	def alphabeta(self, game, time_left, depth, alpha=float("-inf"), beta=float("inf"),maximizing_player=True):
		"""Implementation of the alphabeta algorithm

		Args:
			game (Board): A board and game state.
			time_left (function): Used to determine time left before timeout
			depth: Used to track how deep you are in the search tree
			alpha (float): Alpha value for pruning
			beta (float): Beta value for pruning
			maximizing_player (bool): True if maximizing player is active.

		Returns:
			(tuple,tuple, int): best_move_queen1,best_move_queen2, val
		"""
		# TODO: finish this function!
		raise NotImplementedError
		return best_move_queen1,best_move_queen2, val

