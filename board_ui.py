import pygame
import chess
import os

# Constants
WIDTH, HEIGHT = 600, 600
SQ_SIZE = WIDTH // 8
COLORS = [pygame.Color(64, 64, 64), pygame.Color(48, 48, 48)]

PIECE_NAME_MAP = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king"
}


def _load_pieces():
    pieces = {}
    for color in [chess.WHITE, chess.BLACK]:
        c_prefix = 'w' if color == chess.WHITE else 'b'
        for p_type, p_name in PIECE_NAME_MAP.items():
            filename = f"{c_prefix}-{p_name}.png"
            path = os.path.join("pieces", filename)
            try:
                img = pygame.image.load(path)
                # Using the piece symbol as key (e.g., 'P', 'n')
                piece_obj = chess.Piece(p_type, color)
                pieces[piece_obj.symbol()] = pygame.transform.scale(img, (SQ_SIZE, SQ_SIZE))
            except pygame.error:
                print(f"Warning: Could not load {path}")
    return pieces


class BoardUI:
    def __init__(self, screen):
        self.screen = screen
        self.pieces = _load_pieces()

    def draw(self, board, selected_square, valid_moves):
        pygame.draw.rect(self.screen, (0, 0, 0), pygame.Rect(0, 0, WIDTH, HEIGHT))
        self._draw_squares()
        self._draw_highlights(selected_square, valid_moves)
        self._draw_pieces(board)

    def _draw_squares(self):
        for r in range(8):
            for c in range(8):
                color = COLORS[(r + c) % 2]
                pygame.draw.rect(self.screen, color,
                                 pygame.Rect(c * SQ_SIZE, r * SQ_SIZE, SQ_SIZE, SQ_SIZE))

    def _draw_highlights(self, selected_square, valid_moves):
        s = pygame.Surface((SQ_SIZE, SQ_SIZE), pygame.SRCALPHA)
        if selected_square is not None:
            r, c = 7 - chess.square_rank(selected_square), chess.square_file(selected_square)
            s.fill((0, 0, 255, 100))
            self.screen.blit(s, (c * SQ_SIZE, r * SQ_SIZE))

            for move in valid_moves:
                target_sq = move.to_square
                tr, tc = 7 - chess.square_rank(target_sq), chess.square_file(target_sq)
                s.fill((0, 255, 0, 100))
                self.screen.blit(s, (tc * SQ_SIZE, tr * SQ_SIZE))

    def _draw_pieces(self, board):
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                r, c = 7 - chess.square_rank(square), chess.square_file(square)
                self.screen.blit(self.pieces[piece.symbol()], (c * SQ_SIZE, r * SQ_SIZE))