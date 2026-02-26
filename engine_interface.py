import chess
import numpy as np
import threading
import os

# Global lock for file access to prevent race conditions during self-play
file_lock = threading.Lock()

# Constants for board representation
PIECE_TO_INDEX = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5
}

def board_to_tensor(board: chess.Board, shape=(12, 8, 8)):
    """Converts a chess.Board to a tensor representation of given shape, defaulting to 12x8x8."""
    import torch
    tensor = np.zeros(shape, dtype=np.float32)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row = 7 - chess.square_rank(square)
            col = chess.square_file(square)
            # 0-5 for white pieces, 6-11 for black pieces
            index = PIECE_TO_INDEX[piece.piece_type] + (6 if piece.color == chess.BLACK else 0)
            
            # Ensure index is within bounds of the first dimension
            if index < shape[0]:
                tensor[index, row, col] = 1.0
    return torch.from_numpy(tensor).unsqueeze(0) # Add batch dimension

class BaseEngine:
    def __init__(self, name="Base Engine"):
        self.name = name
        self.game_history = [] # Stores (fen, move_uci)
        self.color = None
        self._move_queue = None
        self._is_thinking = False
        self.training_enabled = True

    def async_get_move(self, board: chess.Board, callback_queue):
        """
        Starts a background thread to compute the move.
        The result will be put into the provided callback_queue.
        """
        if self._is_thinking:
            return

        self._is_thinking = True
        
        def _thread_target():
            try:
                # Copy board to ensure thread safety
                board_copy = board.copy()
                move = self.compute_move(board_copy)
                callback_queue.put(move)
            finally:
                self._is_thinking = False

        thread = threading.Thread(target=_thread_target, daemon=True)
        thread.start()

    def set_color(self, color: chess.Color):
        """Sets the color the engine is playing as."""
        self.color = color

    def compute_move(self, board: chess.Board) -> chess.Move:
        """
        Given the current board state, return the best move.
        This should be overridden by specific engine implementations.
        """
        raise NotImplementedError("Engines must implement compute_move")

    def evaluate_material(self, board: chess.Board, color: chess.Color = None) -> float:
        """
        Simple material evaluation. If color is None, returns absolute score (White - Black).
        If color is provided, returns score relative to that color.
        """
        values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
        score = 0
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece:
                val = values[piece.piece_type]
                score += val if piece.color == chess.WHITE else -val
        
        if color == chess.BLACK:
            return -score
        return score

    def record_move(self, board: chess.Board, move: chess.Move):
        """Records a move in the game history."""
        self.game_history.append((board.fen(), move.uci()))

    def notify_outcome(self, outcome: chess.Outcome) -> bool:
        """
        Called when the game ends.
        Returns: True if the engine wants to reset the game immediately.
        """
        print(f"Engine {self.name} notified of outcome: {outcome.result()}")
        
        if self.game_history:
            self.save_training_data(outcome)

        self.game_history = [] # Clear history for next game
        return False # Default to not resetting

    def get_reward(self, outcome: chess.Outcome) -> float:
        """
        Calculates the reward for the engine based on the game outcome.
        Can be overridden by subclasses to use different weights.
        """
        if outcome.winner is None:
            return 0.0
        return 1.0 if outcome.winner == self.color else -1.0

    def save_training_data(self, outcome: chess.Outcome, max_games=1000, history=None):
        """Saves game history to JSON and model to PTH with file locking."""
        if not self.training_enabled:
            return

        import json
        import torch

        # Determine reward using engine-specific logic
        reward = self.get_reward(outcome)

        dir_path = os.path.join("training_data", self.name.lower().replace(" ", "_"))
        os.makedirs(dir_path, exist_ok=True)
        
        json_path = os.path.join(dir_path, f"{self.name.lower().replace(' ', '_')}.json")
        pth_path = os.path.join(dir_path, f"{self.name.lower().replace(' ', '_')}.pth")

        if history is None:
            history = self.game_history

        game_data = {
            "reward": reward,
            "color": "white" if self.color == chess.WHITE else "black",
            "history": history # Stores [(fen, move_uci), ...]
        }

        with file_lock:
            # Save JSON (rolling window)
            all_data = []
            if os.path.exists(json_path):
                try:
                    with open(json_path, "r") as f:
                        for line in f:
                            if line.strip():
                                all_data.append(json.loads(line))
                except Exception as e:
                    print(f"Error loading training data for saving: {e}")

            all_data.append(game_data)

            if len(all_data) > max_games:
                all_data = all_data[-max_games:]

            with open(json_path, "w") as f:
                for entry in all_data:
                    f.write(json.dumps(entry) + "\n")

        # Save model (PTH)
        self.save_model()
        
        print(f"Training data saved to {json_path}")

    def save_model(self):
        """Saves the engine's model to a PTH file with file locking."""
        import torch
        dir_path = os.path.join("training_data", self.name.lower().replace(" ", "_"))
        os.makedirs(dir_path, exist_ok=True)
        pth_path = os.path.join(dir_path, f"{self.name.lower().replace(' ', '_')}.pth")

        if hasattr(self, 'model'):
            with file_lock:
                torch.save(self.model.state_dict(), pth_path)
                print(f"Model saved to {pth_path}")

    def load_training_data(self):
        """Loads training data from JSON with file locking."""
        import json
        dir_path = os.path.join("training_data", self.name.lower().replace(" ", "_"))
        json_path = os.path.join(dir_path, f"{self.name.lower().replace(' ', '_')}.json")
        
        if not os.path.exists(json_path):
            print(f"No training data found at {json_path}")
            return []

        with file_lock:
            all_data = []
            try:
                with open(json_path, "r") as f:
                    for line in f:
                        if line.strip():
                            all_data.append(json.loads(line))
                return all_data
            except Exception as e:
                print(f"Error loading training data: {e}")
                return []


class RandomEngine(BaseEngine):
    """A simple example engine that picks moves at random for testing."""
    def __init__(self):
        super().__init__("Random Engine")

    def compute_move(self, board: chess.Board) -> chess.Move:
        import random
        color_str = "White" if self.color == chess.WHITE else "Black"
        # Left for debugging
        #print(f"{self.name} ({color_str}) is thinking...")
        move = random.choice(list(board.legal_moves))
        self.record_move(board, move)
        return move

    def notify_outcome(self, outcome: chess.Outcome) -> bool:
        color_str = "White" if self.color == chess.WHITE else "Black"
        print(f"{self.name} ({color_str}) received outcome: {outcome.result()}")
        super().notify_outcome(outcome)
        return True

    def get_reward(self, outcome: chess.Outcome) -> float:
        """Example of engine-specific reward logic, random engines are just happy to play."""
        if outcome.winner is None:
            return 0.1
        return 1.0 if outcome.winner == self.color else -1.0