import importlib.util
import os
import queue
import sys
import threading
import chess
import pygame
import pygame_gui
from board_ui import BoardUI, SQ_SIZE
from engine_interface import BaseEngine, RandomEngine


class ChessGame:
    def __init__(self):
        pygame.init()

        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("PySultan")

        self.manager = pygame_gui.UIManager((800, 600))
        self.ui = BoardUI(self.screen)
        self.board = chess.Board()
        self.board_lock = threading.Lock()

        self.selected_square = None
        self.valid_moves = []
        self.game_over_notified = False
        self.game_running = False

        # Engine Management
        self.available_engines = {"Human": None, "Random Engine": RandomEngine}
        self._load_external_engines()

        self.white_engine = None
        self.black_engine = self.available_engines["Random Engine"]()
        if self.black_engine:
            self.black_engine.training_enabled = True # Default state matches UI

        self.engine_queue = queue.Queue()
        self.is_engine_thinking = False

        self._setup_gui()

    def _setup_gui(self):
        engine_names = list(self.available_engines.keys())

        # White Engine UI
        pygame_gui.elements.UILabel(relative_rect=pygame.Rect((610, 20), (140, 30)),
                                    text="White Player:", manager=self.manager)
        self.white_dropdown = pygame_gui.elements.UIDropDownMenu(
            options_list=engine_names, starting_option="Human",
            relative_rect=pygame.Rect((610, 50), (150, 30)), manager=self.manager)
        self.white_thinking_icon = self._create_thinking_icon((765, 50))

        # Black Engine UI
        pygame_gui.elements.UILabel(relative_rect=pygame.Rect((610, 80), (140, 30)),
                                    text="Black Player:", manager=self.manager)
        self.black_dropdown = pygame_gui.elements.UIDropDownMenu(
            options_list=engine_names, starting_option="Random Engine",
            relative_rect=pygame.Rect((610, 110), (150, 30)), manager=self.manager)
        self.black_thinking_icon = self._create_thinking_icon((765, 130))

        self.training_checkbox = pygame_gui.elements.UICheckBox(
            relative_rect=pygame.Rect((610, 150), (25, 25)),
            text="Training Enabled",
            manager=self.manager
        )
        self.training_checkbox.set_state(True)

        pygame_gui.elements.UILabel(relative_rect=pygame.Rect((610, 180), (180, 30)),
                                    text="Move Log:", manager=self.manager)

        self.move_log_box = pygame_gui.elements.UITextBox(
            html_text="",
            relative_rect=pygame.Rect((610, 210), (180, 270)),
            manager=self.manager
        )

        self.start_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((610, 490), (180, 40)),
            text='Start Game', manager=self.manager)

        self.reset_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((610, 540), (180, 40)),
            text='Reset Game', manager=self.manager)

    def _create_thinking_icon(self, pos, color=(200, 200, 200)):
        """Creates a hidden magnifying glass icon with a specific color."""
        icon_surface = pygame.Surface((30, 30), pygame.SRCALPHA)
        pygame.draw.circle(icon_surface, color, (12, 12), 8, 2)
        pygame.draw.line(icon_surface, color, (18, 18), (25, 25), 3)

        icon = pygame_gui.elements.UIImage(
            relative_rect=pygame.Rect(pos, (30, 30)),
            image_surface=icon_surface,
            manager=self.manager)
        icon.hide()
        return icon

    def update_thinking_icons(self):
        """Update the visibility and color of icons based on turn and check status."""
        is_check = self.board.is_check()
        color = (255, 50, 50) if is_check else (200, 200, 200)

        # We need to recreate the surface to change color dynamically
        def get_colored_surface(c):
            surf = pygame.Surface((30, 30), pygame.SRCALPHA)
            pygame.draw.circle(surf, c, (12, 12), 8, 2)
            pygame.draw.line(surf, c, (18, 18), (25, 25), 3)
            return surf

        # Toggle White Icon
        if self.board.turn == chess.WHITE and self.game_running and not self.board.is_game_over():
            self.white_thinking_icon.set_image(get_colored_surface(color))
            self.white_thinking_icon.show()
        else:
            self.white_thinking_icon.hide()

        # Toggle Black Icon
        if self.board.turn == chess.BLACK and self.game_running and not self.board.is_game_over():
            self.black_thinking_icon.set_image(get_colored_surface(color))
            self.black_thinking_icon.show()
        else:
            self.black_thinking_icon.hide()

    def execute_move(self, move):
        with self.board_lock:
            self.board.push(move)
            self.selected_square = None
            self.valid_moves = []

            move_history = [m.uci() for m in self.board.move_stack]
            formatted_history = ""
            for i in range(0, len(move_history), 2):
                w = move_history[i]
                b = move_history[i+1] if i+1 < len(move_history) else ""
                formatted_history += f"{(i//2)+1}. {w} {b}<br>"

            self.move_log_box.set_text(formatted_history)

            if self.move_log_box.scroll_bar:
                self.move_log_box.scroll_bar.set_scroll_from_start_percentage(1.0)

    def _load_external_engines(self):
        """Scans the engines/ folder for valid engine classes."""
        if not os.path.exists("engines"):
            return

        for filename in os.listdir("engines"):
            if filename.endswith(".py") and filename != "__init__.py":
                module_name = filename[:-3]
                path = os.path.join("engines", filename)
                spec = importlib.util.spec_from_file_location(module_name, path)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)

                for name, obj in mod.__dict__.items():
                    if isinstance(obj, type) and issubclass(obj, BaseEngine) and obj is not BaseEngine:
                        self.available_engines[obj.__name__] = obj  # Store class, not instance

    def check_game_over(self):
        if self.board.is_game_over() and not self.game_over_notified:
            outcome = self.board.outcome()

            should_reset = False
            if self.white_engine and self.white_engine.notify_outcome(outcome):
                should_reset = True
            if self.black_engine and self.black_engine.notify_outcome(outcome):
                should_reset = True

            self.game_over_notified = True

            if should_reset:
                pygame.time.delay(1000)
                self.reset_game()
                self.game_running = True

    def handle_click(self, pos):
        col = pos[0] // SQ_SIZE
        row = 7 - (pos[1] // SQ_SIZE)
        square = chess.square(col, row)

        if self.selected_square is not None:
            piece = self.board.piece_at(self.selected_square)
            move = chess.Move(self.selected_square, square)

            if piece and piece.piece_type == chess.PAWN:
                if (piece.color == chess.WHITE and chess.square_rank(square) == 7) or \
                        (piece.color == chess.BLACK and chess.square_rank(square) == 0):
                    move.promotion = chess.QUEEN

            if move in self.board.legal_moves:
                self.execute_move(move)  # Use the wrapper
                return

        piece = self.board.piece_at(square)
        if piece and piece.color == self.board.turn:
            self.selected_square = square
            self.valid_moves = [m for m in self.board.legal_moves if m.from_square == square]
        else:
            self.selected_square = None
            self.valid_moves = []

    def reset_game(self):
        self.board = chess.Board()
        self.selected_square = None
        self.valid_moves = []
        self.game_over_notified = False
        self.game_running = False
        self.is_engine_thinking = False
        self.white_thinking_icon.hide()
        self.black_thinking_icon.hide()
        while not self.engine_queue.empty():
            self.engine_queue.get()

    def run(self):
        clock = pygame.time.Clock()
        while True:
            time_delta = clock.tick(60) / 1000.0
            self.check_game_over()

            self.update_thinking_icons()

            # Engine Logic
            if self.game_running and not self.board.is_game_over():
                if self.white_engine:
                    self.white_engine.set_color(chess.WHITE)
                if self.black_engine:
                    self.black_engine.set_color(chess.BLACK)

                current_engine = self.white_engine if self.board.turn == chess.WHITE else self.black_engine

                if current_engine is not None:
                    if not self.is_engine_thinking:
                        self.is_engine_thinking = True
                        current_engine.async_get_move(self.board, self.engine_queue)
                    else:
                        try:
                            move = self.engine_queue.get_nowait()
                            self.execute_move(move)
                            self.is_engine_thinking = False
                        except queue.Empty:
                            pass

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame_gui.UI_BUTTON_PRESSED:
                    if event.ui_element == self.start_button:
                        self.game_running = True
                    elif event.ui_element == self.reset_button:
                        self.reset_game()
                        self.is_engine_thinking = False

                # Handle GUI Events
                if event.type in (pygame_gui.UI_CHECK_BOX_CHECKED, pygame_gui.UI_CHECK_BOX_UNCHECKED):
                    if event.ui_element == self.training_checkbox:
                        enabled = self.training_checkbox.get_state()
                        if self.white_engine: self.white_engine.training_enabled = enabled
                        if self.black_engine: self.black_engine.training_enabled = enabled

                if event.type == pygame_gui.UI_DROP_DOWN_MENU_CHANGED:
                    if event.ui_element == self.white_dropdown:
                        engine_cls = self.available_engines[event.text]
                        self.white_engine = engine_cls() if engine_cls else None
                        if self.white_engine:
                            self.white_engine.training_enabled = self.training_checkbox.get_state()
                    elif event.ui_element == self.black_dropdown:
                        engine_cls = self.available_engines[event.text]
                        self.black_engine = engine_cls() if engine_cls else None
                        if self.black_engine:
                            self.black_engine.training_enabled = self.training_checkbox.get_state()

                if event.type == pygame_gui.UI_BUTTON_PRESSED:
                    if event.ui_element == self.reset_button:
                        self.reset_game()

                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    self.reset_game()

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if self.game_running and not self.board.is_game_over():
                        is_human_turn = (self.board.turn == chess.WHITE and self.white_engine is None) or \
                                        (self.board.turn == chess.BLACK and self.black_engine is None)
                        if is_human_turn and event.pos[0] < 600:
                            self.handle_click(event.pos)

                self.manager.process_events(event)

            self.manager.update(time_delta)

            # Rendering
            self.screen.fill((45, 45, 45))  # UI Panel color
            with self.board_lock:
                self.ui.draw(self.board, self.selected_square, self.valid_moves)

            if self.board.is_game_over():
                self.draw_game_over_message()

            self.manager.draw_ui(self.screen)
            pygame.display.flip()

    def draw_game_over_message(self):
        font = pygame.font.SysFont("Arial", 48, bold=True)
        outcome = self.board.outcome()
        result_text = outcome.result() if outcome else "Game Over"

        text_surface = font.render(f"RESULT: {result_text}", True, pygame.Color("red"))
        text_rect = text_surface.get_rect(center=(300, 300))

        # Draw a solid semi-transparent background
        overlay = pygame.Surface((450, 100))
        overlay.set_alpha(200)
        overlay.fill((255, 255, 255))
        self.screen.blit(overlay, (75, 250))
        self.screen.blit(text_surface, text_rect)

        # Helper text
        small_font = pygame.font.SysFont("Arial", 20)
        reset_hint = small_font.render("Press 'R' to Reset", True, pygame.Color("black"))
        self.screen.blit(reset_hint, (240, 320))

if __name__ == "__main__":
    game = ChessGame()
    game.run()