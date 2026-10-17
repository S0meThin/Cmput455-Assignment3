# CMPUT 455 Assignment 3 starter code
# This is a sample solution for assignment 2.
# Implement the new commands to complete the assignment
# Full assignment specification and game rules on Canvas

import ast
import math
import random
import time
from sys import stderr
from typing import Any, Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Assignment restrictions (see assignment page)
# ---------------------------------------------------------------------------
MIN_HEAPS = 1
MAX_HEAPS = 20
MIN_TOKENS = 1
MAX_TOKENS = 20
MIN_VALUE = 1
MAX_VALUE = 20
KOMI_LIMIT = 100  # -100 < komi < 100, and komi must be an integer + 0.5

BLACK = 'b'
WHITE = 'w'
COLORS = (BLACK, WHITE)

Token = Tuple[str, int]
Heap = List[Token]
Heaps = List[Heap]

# ---------------------------------------------------------------------------
# A2: solver constants
# ---------------------------------------------------------------------------
MIN_TIMELIMIT = 1
MAX_TIMELIMIT = 100
DEFAULT_TIMELIMIT = 1

INFINITY = 1 << 30
CHECK_NODES = 1024  # how often the search looks at the clock

EXACT, LOWER, UPPER = 0, 1, 2  # what a transposition table value says

SIZE_BITS = 5  # bits per heap size in a state key, enough for MAX_TOKENS
TOPLAY_BIT = 1 << (SIZE_BITS * MAX_HEAPS)
TT_BITS = 20
# Fibonacci hashing: multiply, keep 64 bits, and use the top ones, so that
# every bit of the key, not just the size of the first few heaps, picks the slot
TT_MULTIPLIER = 0x9E3779B97F4A7C15
TT_MASK64 = (1 << 64) - 1
TT_SHIFT = 64 - TT_BITS
# ---------------------------------------------------------------------------
# A2: end of solver constants
# ---------------------------------------------------------------------------


def not_yet() -> bool:
    raise NotImplementedError("Command not implemented.")
    return False


def print_error(error: str) -> None:
    print(error, file=stderr)


def opponent(color: str) -> str:
    return WHITE if color == BLACK else BLACK


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
def is_int(obj: Any) -> bool:
    # bool is a subclass of int; True/False are not valid values
    return isinstance(obj, int) and not isinstance(obj, bool)


def is_token(obj: Any) -> bool:
    return (isinstance(obj, tuple)
            and len(obj) == 2
            and obj[0] in COLORS
            and is_int(obj[1])
            and MIN_VALUE <= obj[1] <= MAX_VALUE)


def is_heap(obj: Any) -> bool:
    return (isinstance(obj, list)
            and MIN_TOKENS <= len(obj) <= MAX_TOKENS
            and all(is_token(token) for token in obj))


def is_heaps(obj: Any) -> bool:
    return (isinstance(obj, list)
            and MIN_HEAPS <= len(obj) <= MAX_HEAPS
            and all(is_heap(heap) for heap in obj))


def parse_komi(text: str) -> Optional[float]:
    try:
        komi = float(text)
    except ValueError:
        return None
    if not (-KOMI_LIMIT < komi < KOMI_LIMIT):
        return None
    if (komi - 0.5) != int(komi - 0.5):
        return None
    return komi


def parse_heaps(text: str) -> Optional[Heaps]:
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return None
    if not is_heaps(parsed):
        return None
    return parsed


def parse_heap_number(args: str) -> Optional[int]:
    parts = args.split()
    if len(parts) != 1:
        return None
    try:
        return int(parts[0])
    except ValueError:
        return None


def parse_timelimit(args: str) -> Optional[int]:
    parts = args.split()
    if len(parts) != 1:
        return None
    try:
        seconds = int(parts[0])
    except ValueError:
        return None
    if not (MIN_TIMELIMIT <= seconds <= MAX_TIMELIMIT):
        return None
    return seconds

# ---------------------------------------------------------------------------
# Game
# ---------------------------------------------------------------------------
class HeapGo:
    def __init__(self, komi: float, heaps: Heaps) -> None:
        self.komi = komi
        self.heaps = heaps
        self.toplay = BLACK
        self.score = {BLACK: 0, WHITE: komi}

    def __str__(self) -> str:
        return "k {} {}".format(self.komi, self.heaps)

    def is_legal(self, heap_number: int) -> bool:
        return 0 <= heap_number < len(self.heaps) and len(self.heaps[heap_number]) > 0

    def legal_moves(self) -> List[int]:
        return [i for i in range(len(self.heaps)) if self.is_legal(i)]

    def game_over(self) -> bool:
        return len(self.legal_moves()) == 0

    def play(self, heap_number: int) -> bool:
        if not self.is_legal(heap_number):
            return False
        self.play_move(heap_number)  # A2: the rules moved into play_move
        return True

    def winner(self) -> Optional[str]:
        if not self.game_over():
            return None
        if self.score[BLACK] > self.score[WHITE]:
            return BLACK
        if self.score[WHITE] > self.score[BLACK]:
            return WHITE
        return None  # draw; impossible with komi = integer + 0.5

# ============================================================================
# A2: play_move / undo_move / copy
# ============================================================================
    def play_move(self, heap_number: int) -> Tuple[Heap, int]:
        """Play a legal move.  Returns what is needed to undo it."""
        heap = self.heaps[heap_number]
        taken = []
        # Rule 1: take all own-color tokens from the top
        while heap and heap[-1][0] == self.toplay:
            taken.append(heap.pop())
        # Rule 2: then one opponent token, if any remain
        if heap:
            taken.append(heap.pop())
        gained = sum(value for _, value in taken)
        self.score[self.toplay] += gained
        self.toplay = opponent(self.toplay)
        return taken, gained

    def undo_move(self, heap_number: int, taken: Heap, gained: int) -> None:
        self.toplay = opponent(self.toplay)
        self.score[self.toplay] -= gained
        self.heaps[heap_number].extend(reversed(taken))

    def copy(self) -> "HeapGo":
        game = HeapGo(self.komi, [list(heap) for heap in self.heaps])
        game.toplay = self.toplay
        game.score = dict(self.score)
        return game
# ============================================================================
# A2: end of game changes
# ============================================================================


# ============================================================================
# A2: the solver
# ============================================================================
class Timeout(Exception):
    pass


class Solver:
    """Negamax with alpha-beta pruning, move ordering and a transposition
    table, on the score difference.  The value of a position is the best
    reachable (points for the player to move) - (points for the opponent),
    counting only the points still to be won.  The player to move wins when
    the score difference so far plus that value is positive."""

    def __init__(self) -> None:
        self.tt: List[Optional[Tuple[int, int, int, int]]] = [None] * (1 << TT_BITS)
        self.deadline = 0.0
        self.nodes = 0

    def new_game(self) -> None:
        """Table entries are keyed by heap sizes, so they only stay valid
        while the heaps themselves do not change."""
        self.tt = [None] * (1 << TT_BITS)

    def prepare(self, game: HeapGo) -> None:
        """Precompute what a move takes, for every color, heap and heap size.
        The search then never has to look at a token again."""
        self.num_heaps = len(game.heaps)
        self.gain: List[List[List[int]]] = [[], []]
        self.take: List[List[List[int]]] = [[], []]
        for color_index, color in enumerate(COLORS):
            for heap in game.heaps:
                gains, takes = [0], [0]  # nothing to take from an empty heap
                for size in range(1, len(heap) + 1):
                    taken, gained = 0, 0
                    while taken < size and heap[size - 1 - taken][0] == color:
                        gained += heap[size - 1 - taken][1]
                        taken += 1
                    if taken < size:
                        gained += heap[size - 1 - taken][1]
                        taken += 1
                    gains.append(gained)
                    takes.append(taken)
                self.gain[color_index].append(gains)
                self.take[color_index].append(takes)

    def solve(self, game: HeapGo, timelimit: int) -> Optional[Tuple[str, Optional[int]]]:
        """Returns (winner, move), or None when the time limit runs out.
        The move is the one to play when the player to move wins, else None."""
        self.nodes = 0
        self.deadline = time.time() + timelimit
        difference = game.score[game.toplay] - game.score[opponent(game.toplay)]
        if game.game_over():
            value, move = 0, None
        else:
            try:
                value, move = self.root(game, difference)
            except Timeout:
                return None
        if difference + value > 0:
            return game.toplay, move
        return opponent(game.toplay), None

    def root(self, game: HeapGo, difference: float) -> Tuple[int, int]:
        self.prepare(game)
        sizes = [len(heap) for heap in game.heaps]
        key = sum(size << (SIZE_BITS * i) for i, size in enumerate(sizes))
        if game.toplay == WHITE:
            key |= TOPLAY_BIT
        # All that solve has to report is who wins, so search with the
        # narrowest window that still answers that question.
        threshold = math.ceil(-difference)
        alpha, beta = threshold - 1, threshold
        toplay = COLORS.index(game.toplay)
        gain, take = self.gain[toplay], self.take[toplay]
        best, best_move = -INFINITY, -1
        for move in self.moves(sizes, toplay, -1):
            size = sizes[move]
            taken, gained = take[move][size], gain[move][size]
            sizes[move] = size - taken
            value = gained - self.search(
                sizes, 1 - toplay, (key ^ TOPLAY_BIT) - (taken << (SIZE_BITS * move)),
                gained - beta, gained - alpha)
            sizes[move] = size
            if value > best:
                best, best_move = value, move
                if value > alpha:
                    alpha = value
                    if alpha >= beta:
                        break
        return best, best_move

    def search(self, sizes: List[int], toplay: int, key: int,
               alpha: int, beta: int) -> int:
        self.nodes += 1
        if self.nodes % CHECK_NODES == 0 and time.time() > self.deadline:
            raise Timeout
        alpha_original = alpha
        tt_move = -1
        slot = ((key * TT_MULTIPLIER) & TT_MASK64) >> TT_SHIFT
        entry = self.tt[slot]
        if entry is not None and entry[0] == key:
            value, flag, tt_move = entry[1], entry[2], entry[3]
            if flag == EXACT:
                return value
            if flag == LOWER:
                if value > alpha:
                    alpha = value
            elif value < beta:
                beta = value
            if alpha >= beta:
                return value
        moves = self.moves(sizes, toplay, tt_move)
        if not moves:
            return 0
        gain, take = self.gain[toplay], self.take[toplay]
        child_toplay, child_key = 1 - toplay, key ^ TOPLAY_BIT
        best, best_move = -INFINITY, moves[0]
        for move in moves:
            size = sizes[move]
            taken, gained = take[move][size], gain[move][size]
            sizes[move] = size - taken
            value = gained - self.search(
                sizes, child_toplay, child_key - (taken << (SIZE_BITS * move)),
                gained - beta, gained - alpha)
            sizes[move] = size
            if value > best:
                best, best_move = value, move
                if value > alpha:
                    alpha = value
                    if alpha >= beta:
                        break
        if best <= alpha_original:
            flag = UPPER
        elif best >= beta:
            flag = LOWER
        else:
            flag = EXACT
        self.tt[slot] = (key, best, flag, best_move)
        return best

    def moves(self, sizes: List[int], toplay: int, tt_move: int) -> List[int]:
        """Legal moves, best first: the move from the table, then the moves
        that win the most points now."""
        moves = [i for i in range(self.num_heaps) if sizes[i]]
        if len(moves) < 2:
            return moves
        gain = self.gain[toplay]
        moves.sort(key=lambda move: -gain[move][sizes[move]])
        if tt_move >= 0 and tt_move != moves[0] and sizes[tt_move]:
            moves.remove(tt_move)
            moves.insert(0, tt_move)
        return moves

# ============================================================================
# A2: end of the solver
# ============================================================================


CommandMap = Dict[str, Callable[[str], bool]]


class CommandInterface:
    def __init__(self) -> None:
        self.game: Optional[HeapGo] = None
        self.solver = Solver()  # A2
        self.timelimit = DEFAULT_TIMELIMIT  # A2
        # you can add your own initialisation here
        self.commands: CommandMap = {
            "help": self.cmd_help,
            "heapgo": self.cmd_heapgo,
            "show": self.cmd_show,
            "toplay": self.cmd_toplay,
            "play": self.cmd_play,
            "legal": self.cmd_legal,
            "genmove": self.cmd_genmove,
            "score": self.cmd_score,
            "winner": self.cmd_winner,
            "solve": self.cmd_solve,  # A2
            "timelimit": self.cmd_timelimit,  # A2
            "simulate": self.cmd_simulate,
            "numsimulations": self.cmd_numsimulations,
            "policy": self.cmd_policy,
            "policymoves": self.cmd_policymoves,
            }

#============================================================================
# Command implementations
#============================================================================
    def cmd_heapgo(self, args: str) -> bool:
        parts = args.split(maxsplit=1)
        if len(parts) != 2:
            print_error("usage: heapgo komi game")
            return False
        komi = parse_komi(parts[0])
        if komi is None:
            print_error("invalid komi: {}".format(parts[0]))
            return False
        heaps = parse_heaps(parts[1])
        if heaps is None:
            print_error("invalid game: {}".format(parts[1]))
            return False
        self.game = HeapGo(komi, heaps)
        self.solver.new_game()  # A2
        return True

    def cmd_show(self, args: str) -> bool:
        if self.game is None:
            print_error("no game started")
            return False
        print(self.game)
        return True

    def cmd_toplay(self, args: str) -> bool:
        if self.game is None:
            print_error("no game started")
            return False
        if args not in COLORS:
            print_error("invalid color: {}".format(args))
            return False
        self.game.toplay = args
        return True

    def cmd_play(self, args: str) -> bool:
        if self.game is None:
            print_error("no game started")
            return False
        heap_number = parse_heap_number(args)
        if heap_number is None:
            print_error("invalid heap number: {}".format(args))
            return False
        if not self.game.play(heap_number):
            print_error("illegal move: {}".format(heap_number))
            return False
        return True

    def cmd_legal(self, args: str) -> bool:
        if self.game is None:
            print_error("no game started")
            return False
        heap_number = parse_heap_number(args)
        if heap_number is None:
            print_error("invalid heap number: {}".format(args))
            return False
        print("yes" if self.game.is_legal(heap_number) else "no")
        return True

    def cmd_score(self, args: str) -> bool:
        if self.game is None:
            print_error("no game started")
            return False
        print("b {} w {}".format(self.game.score[BLACK], self.game.score[WHITE]))
        return True

    def cmd_winner(self, args: str) -> bool:
        if self.game is None:
            print_error("no game started")
            return False
        winner = self.game.winner()
        if winner is None:
            print_error("game not over" if not self.game.game_over() else "draw")
            return False
        print(winner)
        return True

    def cmd_solve(self, args: str) -> bool:
        if self.game is None:
            print_error("no game started")
            return False
        result = self.solver.solve(self.game, self.timelimit)
        if result is None:
            print_error("unsolved within the time limit of {}s".format(self.timelimit))
            return False
        winner, move = result
        print(winner if move is None else "{} {}".format(winner, move))
        return True

    def cmd_timelimit(self, args: str) -> bool:
        seconds = parse_timelimit(args)
        if seconds is None:
            print_error("invalid time limit: {}".format(args))
            return False
        self.timelimit = seconds
        return True

#============================================================================
# You need to implement or modify the following methods.
#============================================================================
    def cmd_genmove(self, args: str) -> bool:
        if self.game is None:
            print_error("no game started")
            return False
        moves = self.game.legal_moves()
        if not moves:
            print_error("no legal moves")
            return False
        # you need to replace this random move by a move from your
        # simulation-based player
        move = random.choice(moves)
        self.game.play(move)
        print(move)
        return True

    def cmd_simulate(self, args: str) -> bool:
        return not_yet()

    def cmd_numsimulations(self, args: str) -> bool:
        return not_yet()

    def cmd_policy(self, args: str) -> bool:
        return not_yet()

    def cmd_policymoves(self, args: str) -> bool:
        return not_yet()
#============================================================================
# End of functions requiring implementation
#============================================================================

#============================================================================
# Command processing
#============================================================================
    def cmd_help(self, ignore_args: str) -> bool:
        print("\nKnown commands:")
        for cmd in self.commands:
            print(cmd)
        return True

    def process_command(self, cmd_name: str, cmd_args: str) -> None:
        status = "= -1"
        cmd = self.commands.get(cmd_name)
        if cmd:
            try:
                if cmd(cmd_args):
                    status = "= 1"
            except Exception as e:
                print_error(f"Command {cmd_name} with arguments {cmd_args} failed with exception: {e}")
        else:
            print_error("Unknown command. Type 'help' for commands.")
        print(status)

    def main_loop(self) -> None:
        process_commands = True
        while process_commands:
            try:
                line = input()
            except EOFError:
                break
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(maxsplit=1)
            cmd_name = parts[0]
            if cmd_name == "exit":
                process_commands = False
                continue
            cmd_args = parts[1] if len(parts) > 1 else ""
            self.process_command(cmd_name, cmd_args)


if __name__ == "__main__":
    interface = CommandInterface()
    interface.main_loop()
