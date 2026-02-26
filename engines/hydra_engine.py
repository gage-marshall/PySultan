import torch
import torch.nn as nn
import torch.nn.functional as F
import chess
import numpy as np
import os
import threading
from engine_interface import BaseEngine, board_to_tensor

class HydraNet(nn.Module):
    def __init__(self):
        super(HydraNet, self).__init__()
        
        self.conv1 = nn.Conv2d(28, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        # Policy head
        self.policy_conv = nn.Conv2d(128, 73, kernel_size=1)
        
        # Value head
        self.value_conv = nn.Conv2d(128, 1, kernel_size=1)
        self.value_fc1 = nn.Linear(1 * 8 * 8, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Policy: [batch, 73, 8, 8]
        p = self.policy_conv(x)
        
        # Value: [batch, 1]
        v = F.relu(self.value_conv(x))
        v = v.view(-1, 1 * 8 * 8)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))
        
        return p, v

class PyTorchHydra(BaseEngine):
    def __init__(self):
        super().__init__("PyTorch Hydra")
        self.model = HydraNet()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Engine {self.name} is using device: {self.device}")
        self.model.to(self.device)
        self.board_history = []
        self.model_lock = threading.Lock()

        dir_path = os.path.join("training_data", self.name.lower().replace(" ", "_"))
        pth_path = os.path.join(dir_path, f"{self.name.lower().replace(' ', '_')}.pth")
        if os.path.exists(pth_path):
            try:
                self.model.load_state_dict(torch.load(pth_path, map_location=self.device))
                print(f"Loaded existing model from {pth_path}")
            except Exception as e:
                print(f"Failed to load model (probably architecture change): {e}")
                print("Starting with a fresh model.")
        
        self.model.eval()

    def _get_board_state_plane(self, board):
        """Returns a single 8x8 plane representing the piece presence."""
        plane = np.zeros((8, 8), dtype=np.float32)
        for square in chess.SQUARES:
            if board.piece_at(square):
                row = 7 - chess.square_rank(square)
                col = chess.square_file(square)
                plane[row, col] = 1.0
        return plane

    def _encode_board(self, board):
        # 28 planes: 12 pieces + 1 side + 4 castling + 8 EP + 3 history
        tensor = np.zeros((28, 8, 8), dtype=np.float32)
        
        us = board.turn
        them = not us
        
        # 0-5: Our Pieces, 6-11: Their Pieces
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                # If we are black, flip the row for the tensor
                row = 7 - chess.square_rank(square)
                if us == chess.BLACK:
                    row = 7 - row
                
                col = chess.square_file(square)
                
                if piece.color == us:
                    idx = piece.piece_type - 1
                else:
                    idx = 6 + (piece.piece_type - 1)
                    
                tensor[idx, row, col] = 1.0
        
        # 12: Side to move (1.0 for White, 0.0 for Black)
        if us == chess.WHITE:
            tensor[12, :, :] = 1.0
        else:
            tensor[12, :, :] = 0.0
            
        # 13-16: Castling rights (Normalized to US then THEM)
        if board.has_queenside_castling_rights(us): tensor[13, :, :] = 1.0
        if board.has_kingside_castling_rights(us): tensor[14, :, :] = 1.0
        if board.has_queenside_castling_rights(them): tensor[15, :, :] = 1.0
        if board.has_kingside_castling_rights(them): tensor[16, :, :] = 1.0
        
        # 17-24: En passant file (8 planes one-hot)
        if board.ep_square:
            ep_file = chess.square_file(board.ep_square)
            # If black to move, the board is flipped, but files are the same.
            tensor[17 + ep_file, :, :] = 1.0
            
        # 25-27: Last 3 positions (should also be flipped if we are black)
        for i, plane in enumerate(self.board_history[-3:]):
            if us == chess.BLACK:
                tensor[25 + i, :, :] = np.flipud(plane)
            else:
                tensor[25 + i, :, :] = plane
        
        return torch.from_numpy(tensor).unsqueeze(0).to(self.device)

    def _move_to_plane_index(self, board, move):
        """
        Maps a chess.Move to a (plane, row, col) index.
        - 0-55: Queen-like moves
        - 56-63: Knight moves
        - 64-72: Under-promotions
        """
        us = board.turn
        from_sq = move.from_square
        to_sq = move.to_square
        
        # Get perspective row/col for the 'from' square
        from_row = 7 - chess.square_rank(from_sq)
        if us == chess.BLACK:
            from_row = 7 - from_row
        from_col = chess.square_file(from_sq)
        
        # Get relative displacement
        # In our perspective, White moves "up" (+rank)
        r1, c1 = chess.square_rank(from_sq), chess.square_file(from_sq)
        r2, c2 = chess.square_rank(to_sq), chess.square_file(to_sq)
        
        dr = r2 - r1
        dc = c2 - c1
        
        if us == chess.BLACK:
            dr = -dr
            dc = -dc
            dc = dc # no-op for clarity
            
        # 1. Under-promotions
        if move.promotion and move.promotion != chess.QUEEN:
            # 64-66: Rook, 67-69: Bishop, 70-72: Knight
            piece_idx = {chess.ROOK: 0, chess.BISHOP: 1, chess.KNIGHT: 2}[move.promotion]
            # Direction: dc is -1 (left-cap), 0 (straight), 1 (right-cap)
            direction_idx = dc + 1
            plane = 64 + piece_idx * 3 + direction_idx
            return plane, from_row, from_col

        # 2. Knight moves
        knight_offsets = [
            (2, 1), (1, 2), (-1, 2), (-2, 1), 
            (-2, -1), (-1, -2), (1, -2), (2, -1)
        ]
        if (dr, dc) in knight_offsets:
            plane = 56 + knight_offsets.index((dr, dc))
            return plane, from_row, from_col

        # 3. Queen-like moves
        queen_directions = [
            (1, 0), (1, 1), (0, 1), (-1, 1), 
            (-1, 0), (-1, -1), (0, -1), (1, -1)
        ]
        
        # Find direction and distance
        abs_dr, abs_dc = abs(dr), abs(dc)
        distance = max(abs_dr, abs_dc)
        
        if distance > 0:
            unit_dr = dr // distance if abs_dr == distance or abs_dr == 0 else 0
            unit_dc = dc // distance if abs_dc == distance or abs_dc == 0 else 0
            
            # Verify it's a valid queen direction
            if (unit_dr, unit_dc) in queen_directions:
                direction_idx = queen_directions.index((unit_dr, unit_dc))
                plane = direction_idx * 7 + (distance - 1)
                if plane < 56:
                    return plane, from_row, from_col

        return None

    def _evaluate(self, board):
        """
        Leaf evaluation using a blended scoring model
        """
        # 1. Base NN Evaluation
        with self.model_lock:
            self.model.eval()
            with torch.no_grad():
                input_tensor = self._encode_board(board)
                _, value_tensor = self.model(input_tensor)
                nn_score = value_tensor.item() # Perspective: side to move (due to _encode_board)

        # 2. Material Balance (scaled to roughly match NN tanh range)
        material_score = self.evaluate_material(board, board.turn) / 10.0
        
        # Blend NN and Material (50/50)
        score = 0.5 * nn_score + 0.5 * material_score
        
        # 3. Check Awareness
        # board.is_check() means the current player is in check (it's their turn)
        if board.is_check():
            score -= 0.1

        # 4. Repetition Penalty
        if material_score > 0 and any(board.is_repetition(count) for count in [2, 3]):
            score -= 0.1
            
        return score

    def _get_ordered_moves(self, board, policy_prior=None):
        """
        1. Checks
        2. Captures (MVV-LVA)
        3. Promotions
        4. NN Policy Prior
        """
        legal_moves = list(board.legal_moves)
        
        def move_score(move):
            score = 0
            # Check awareness
            if board.gives_check(move):
                score += 100
                
            # MVV-LVA (Most Valuable Victim - Least Valuable Aggressor)
            if board.is_capture(move):
                victim = board.piece_at(move.to_square)
                aggressor = board.piece_at(move.from_square)
                if victim and aggressor:
                    score += 50 + (victim.piece_type * 10) - aggressor.piece_type
            
            # Promotions
            if move.promotion:
                score += 40
                
            # NN Policy Prior
            if policy_prior is not None:
                mapping = self._move_to_plane_index(board, move)
                if mapping:
                    plane, row, col = mapping
                    score += policy_prior[plane, row, col] * 20 # Weight policy prior

            return score

        return sorted(legal_moves, key=move_score, reverse=True)

    def _negamax(self, board, depth, alpha, beta):
        """Standard NegaMax with Alpha-Beta pruning."""
        if depth == 0 or board.is_game_over():
            return self._evaluate(board)

        max_score = -float('inf')
        
        # Move ordering
        policy_prior = None
        if depth > 0: # Only compute policy if we are going to explore children
            with self.model_lock:
                self.model.eval()
                with torch.no_grad():
                    input_tensor = self._encode_board(board)
                    policy_logits, _ = self.model(input_tensor)
                    # Reshape to [73, 8, 8] and apply softmax
                    policy_logits = policy_logits.view(73, 8, 8)
                    policy_prior = torch.softmax(policy_logits.flatten(), dim=0).view(73, 8, 8).cpu().numpy()

        ordered_moves = self._get_ordered_moves(board, policy_prior=policy_prior)
        
        for move in ordered_moves:
            board.push(move)
            score = -self._negamax(board, depth - 1, -beta, -alpha)
            board.pop()
            
            max_score = max(max_score, score)
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        return max_score

    def compute_move(self, board: chess.Board) -> chess.Move:
        # Update board history
        current_plane = self._get_board_state_plane(board)
        self.board_history.append(current_plane)
        if len(self.board_history) > 3:
            self.board_history = self.board_history[-3:]

        # Alpha-Beta Search (Depth 2 for tactical filter)
        best_move = None
        max_score = -float('inf')
        alpha = -float('inf')
        beta = float('inf')
        
        # Get Policy Prior for root ordering
        with self.model_lock:
            self.model.eval()
            with torch.no_grad():
                input_tensor = self._encode_board(board)
                policy_logits, _ = self.model(input_tensor)
                # Reshape to [73, 8, 8] and apply softmax
                policy_logits = policy_logits.view(73, 8, 8)
                probs = torch.softmax(policy_logits.flatten(), dim=0).view(73, 8, 8).cpu().numpy()
        
        ordered_moves = self._get_ordered_moves(board, policy_prior=probs)
        
        move_scores = {}
        for move in ordered_moves:
            board.push(move)
            score = -self._negamax(board, 1, -beta, -alpha)
            board.pop()
            
            move_scores[move.uci()] = score
            
            if score > max_score:
                max_score = score
                best_move = move
            
            alpha = max(alpha, score)
            
        if best_move is None:
            best_move = list(board.legal_moves)[0]
            
        # Record move with search distribution (scores)
        self.record_move(board, best_move, move_scores=move_scores)
        return best_move

    def record_move(self, board: chess.Board, move: chess.Move, move_scores=None):
        """Records the current board state, the move played, and optionally the search scores."""
        # history stores [(fen, best_move_uci, move_scores), ...]
        self.game_history.append((board.fen(), move.uci(), move_scores))

    def get_reward(self, outcome: chess.Outcome) -> float:
        """
        Final state rewards: 
        +1 win, -1 loss, 
        -0.05 draw if material high, +0.05 draw if material low.
        """
        if outcome.winner is not None:
            return 1.0 if outcome.winner == self.color else -1.0
        
        # It's a draw. Check material.
        if not self.game_history:
             return 0.0

        last_fen = self.game_history[-1][0]
        board = chess.Board(last_fen)
        material_balance = self.evaluate_material(board, self.color)
        
        if material_balance > 0: # We were ahead
            return -0.05
        elif material_balance < 0: # We were behind
            return 0.05
        return 0.0

    def notify_outcome(self, outcome: chess.Outcome) -> bool:
        """Called when the game ends. Saves data and triggers training in the background if enabled."""
        print(f"Engine {self.name} notified of outcome: {outcome.result()}")
        
        # Capture history for background thread to avoid blocking the UI
        history_copy = list(self.game_history)
        self.game_history = [] 
        
        if not self.training_enabled:
            return True

        def _background_work():
            if history_copy:
                self.save_training_data(outcome, history=history_copy)
                self.train()
                # Model is saved inside save_training_data and at the end of train()

        threading.Thread(target=_background_work, daemon=True).start()
        
        return True # Auto-reset for self-play training

    def train(self, batch_size=32, epochs=1):
        """Simple training loop to update the network based on search distributions."""
        data = self.load_training_data()
        if not data:
            return

        print(f"Engine {self.name} starting training on {len(data)} games...")
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        for epoch in range(epochs):
            # Sample a batch of games
            import random
            batch_games = random.sample(data, min(len(data), batch_size))
            
            for game in batch_games:
                # game["reward"] is from the perspective of self.color
                game_reward = game["reward"]
                history = game["history"]
                
                # Training on each position in the game
                for entry in history:
                    # Backward compatibility for old training data
                    if len(entry) == 2:
                        fen, move_uci = entry
                        move_scores = None
                    else:
                        fen, move_uci, move_scores = entry

                    board = chess.Board(fen)
                    
                    # Target reward for this specific position's turn
                    pos_reward = game_reward if board.turn == (chess.WHITE if game["color"] == "white" else chess.BLACK) else -game_reward
                    reward_t = torch.tensor([pos_reward], dtype=torch.float32).to(self.device)

                    input_tensor = self._encode_board(board)
                    
                    # Determine target policy
                    if move_scores:
                        # Convert scores to probabilities using softmax with temperature
                        temp = 1.0
                        scores_list = []
                        indices_list = []
                        
                        for uci, score in move_scores.items():
                            m = chess.Move.from_uci(uci)
                            mapping = self._move_to_plane_index(board, m)
                            if mapping:
                                plane, row, col = mapping
                                idx = plane * 64 + row * 8 + col
                                scores_list.append(score / temp)
                                indices_list.append(idx)
                        
                        if scores_list:
                            # Use soft targets
                            target_probs = torch.zeros(73 * 8 * 8).to(self.device)
                            scores_t = torch.tensor(scores_list, dtype=torch.float32).to(self.device)
                            probs_t = torch.softmax(scores_t, dim=0)
                            for i, idx in enumerate(indices_list):
                                target_probs[idx] = probs_t[i]
                        else:
                            move_scores = None # Fallback to one-hot if no valid scores

                    if not move_scores:
                        # Fallback to one-hot encoding of the played move
                        move = chess.Move.from_uci(move_uci)
                        mapping = self._move_to_plane_index(board, move)
                        if mapping:
                            plane, row, col = mapping
                            move_idx = plane * 64 + row * 8 + col
                            target_probs = torch.zeros(73 * 8 * 8).to(self.device)
                            target_probs[move_idx] = 1.0
                        else:
                            continue # Skip invalid moves

                    with self.model_lock:
                        self.model.train()
                        optimizer.zero_grad()
                        policy_pred, value_pred = self.model(input_tensor)
                        
                        # Value loss (outcome)
                        value_loss = F.mse_loss(value_pred, reward_t.unsqueeze(0))
                        
                        # Policy loss (cross-entropy with soft targets)
                        # Flatten policy_pred to [1, 73*8*8]
                        policy_pred = policy_pred.view(1, -1)
                        policy_loss = F.cross_entropy(policy_pred, target_probs.unsqueeze(0))
                        
                        loss = value_loss + policy_loss
                        loss.backward()
                        optimizer.step()
        
        with self.model_lock:
            self.model.eval()
        self.save_model()
        print(f"Engine {self.name} training complete.")
