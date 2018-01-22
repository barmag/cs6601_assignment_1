#!/usr/bin/env python
from isolation import Board, game_as_text
from random import randint
import random
import time


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
        if maximizing_player_turn:
            player_q1, player_q2 = game.get_legal_moves().values()
            opponent_q1, opponent_q2 = game.get_opponent_moves().values()
        else:
            player_q1, player_q2 = game.get_opponent_moves().values()
            opponent_q1, opponent_q2 = game.get_legal_moves().values()
        player_unique_moves = player_q1 + list(set(player_q2) - set(player_q1))
        player_moves = len(player_unique_moves)
        opponent_unique_moves = opponent_q1 + list(set(opponent_q2) - set(opponent_q1))
        opponent_moves = len(opponent_unique_moves)
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
        if maximizing_player_turn:
            # player_q1, player_q2 = game.get_legal_moves().values()
            opponent_q1, opponent_q2 = game.get_opponent_moves().values()
        else:
            # player_q1, player_q2 = game.get_opponent_moves().values()
            opponent_q1, opponent_q2 = game.get_legal_moves().values()
        # player_unique_moves = player_q1 + list(set(player_q2) - set(player_q1))
        # player_moves = len(player_unique_moves)
        opponent_unique_moves = opponent_q1 + list(set(opponent_q2) - set(opponent_q1))
        opponent_moves = len(opponent_unique_moves)
        return float(-opponent_moves)
        # return float(player_moves - opponent_moves)


class TimeoutException(Exception):
    "Thrown when time is out to aid with iterative deepining"
    pass


class CustomPlayer:
    # TODO: finish this class!
    """Player that chooses a move using
    your evaluation function and
    a minimax algorithm
    with alpha-beta pruning.
    You must finish and test this player
    to make sure it properly uses minimax
    and alpha-beta to return a good move."""

    def __init__(self, search_depth=4, eval_fn=OpenMoveEvalFn(), useMiniMax=False):
        """Initializes your player.

        if you find yourself with a superior eval function, update the default
        value of `eval_fn` to `CustomEvalFn()`

        Args:
            search_depth (int): The depth to which your agent will search
            eval_fn (function): Utility function used by your agent
        """
        self.eval_fn = eval_fn
        self.search_depth = search_depth
        self.alpha = - float("inf")
        self.beta = float("inf")
        self.time_threshold = 100
        self.useMiniMax = useMiniMax

    def move(self, game, legal_moves, time_left):
        if self.useMiniMax:
            best_move_queen1, best_move_queen2, utility = self.minimax(game, time_left, 1)
        else:
            best_move_queen1, best_move_queen2, utility = self.alphabeta(game, time_left, depth=self.search_depth)

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
        moves_q1, moves_q2 = game.get_legal_moves().values()
        if len(moves_q1) > 0:
            best_move_queen1 = moves_q1[0]
        else:
            return None, None, -float("inf")
        if len(moves_q2) > 0:
            best_move_queen2 = moves_q2[0]
        else:
            return None, None, -float("inf")
        # return best_move_queen1, best_move_queen2, 0
        all_moves = self.combine_moves(moves_q1, moves_q2)
        # start = time.time()
        v = -float("inf")
        for m1, m2 in all_moves:
            if time_left() < 1.5:
                break
            vx = self.min_minmax(game.forecast_move(m1, m2), time_left)
            if vx > v:
                v = vx
                best_move_queen1, best_move_queen2 = m1, m2

        # print time_left()
        # best_move_queen1, best_move_queen2 = max(all_moves, key=lambda m: self.min_minmax(game.forecast_move(m[0], m[1])))
        # print time.time() - start
        return best_move_queen1, best_move_queen2, v

    def min_minmax(self, game, time_left):
        # just start with one level to test first submission
        moves_q1, moves_q2 = game.get_legal_moves().values()
        if len(moves_q1) > 0:
            best_move_queen1 = moves_q1[0]
        else:
            return -float("inf")
        if len(moves_q2) > 0:
            best_move_queen1 = moves_q2[0]
        else:
            return -float("inf")
        # return self.utility(game, True)
        random_sample = False  # len(moves_q1) * len(moves_q2) > 50
        all_moves = self.combine_moves(moves_q1, moves_q2, random_sample)
        # h_values = {m: self.utility(game.forecast_move(m[0], m[1]), True) for m in all_moves}
        v = float("inf")
        for m1, m2 in all_moves:
            if time_left() < 1.5:
                return v
            vx = self.utility(game.forecast_move(m1, m2), True)
            if vx < v:
                v = vx

        # v = min(h_values.values())
        return v

    def combine_moves(self, q1_moves, q2_moves, random_sample=False):
        # factor = 4 if random_sample else 1
        # all_moves = [(q1, q2) for q1 in q1_moves for q2 in q2_moves if (q1 != q2 and randint(0, 24)%factor == 0)]
        all_moves = [(q1, q2) for q1 in q1_moves for q2 in q2_moves if (q1 != q2)]
        if self.useMiniMax:
            random.shuffle(all_moves)
        return all_moves

    def alphabeta(self, game, time_left, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
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
        start = time.time()
        v, self.alpha, self.beta = -float("inf"), -float("inf"), float("inf")
        for i in range(1, 10):
            self.alpha, self.beta = -float("inf"), float("inf")
            try:
                vx, best_move_queen1, best_move_queen2 = self.max_ab(game, i, time_left, self.alpha, self.beta)
            except TimeoutException:
                print "timed out at level: " + str(i)
                break

        # print time_left()
        # best_move_queen1, best_move_queen2 = max(all_moves, key=lambda m: self.min_minmax(game.forecast_move(m[0], m[1])))
        print time.time() - start
        return best_move_queen1, best_move_queen2, vx

    def max_ab(self, game, depth, time_left, alpha, beta):
        moves_q1, moves_q2 = game.get_legal_moves().values()
        if len(moves_q1) > 0:
            best_move_queen1 = moves_q1[0]
        else:
            return -float("inf"), None, None
        if len(moves_q2) > 0:
            best_move_queen2 = moves_q2[0]
        else:
            return -float("inf"), None, None

        v = -float("inf")
        all_actions = self.combine_moves(moves_q1, moves_q2)
        #    return self.utility(game, True), best_move_queen1, best_move_queen2
        for m1, m2 in all_actions:
            if time_left() < self.time_threshold:
                # return v, best_move_queen1, best_move_queen2
                raise TimeoutException()
            if m1 == m2 and best_move_queen1 is not None:
                pass
            if depth < 0:
                v_r = self.utility(game.forecast_move(m1, m2), True)
            else:
                v_r = self.min_ab(game.forecast_move(m1, m2), depth - 1, time_left, self.alpha, self.beta)

            # v = max(v, v_r)
            if v_r >= v:
                best_move_queen1, best_move_queen2, v = m1, m2, v_r
            if v >= self.beta:
                return v, best_move_queen1, best_move_queen2
            self.alpha = max(self.alpha, v)
        return v, best_move_queen1, best_move_queen2

    def min_ab(self, game, depth, time_left, alpha, beta):
        moves_q1, moves_q2 = game.get_legal_moves().values()
        if len(moves_q1) > 0:
            best_move_queen1 = moves_q1[0]
        else:
            return float("inf")
        if len(moves_q2) > 0:
            best_move_queen2 = moves_q2[0]
        else:
            return float("inf")

        v = float("inf")
        all_actions = self.combine_moves(moves_q1, moves_q2)

        #    return self.utility(game, True)
        for m1, m2 in all_actions:
            if time_left() < self.time_threshold:
                # return v
                raise TimeoutException()
            if depth < 0:
                v_r = self.utility(game.forecast_move(m1, m2), True)
            else:
                v_r, _, _ = self.max_ab(game.forecast_move(m1, m2), depth - 1, time_left, self.alpha, self.beta)
            v = min(v, v_r)
            if v <= alpha:
                return v
            self.beta = min(self.beta, v)
        return v
