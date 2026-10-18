# PySultan

A Python chess interface for human vs AI or AI vs AI play, with a small base class for plugging in your own engines. Built with Pygame and the python-chess library.

## Features

- Interactive chess board with piece movement and legal move highlighting
- Support for both human and AI players (White and Black)
- Extensible engine system - easily add your own AI engines
- **PyTorch Hydra** - Example neural network engine with self-play training
- Built-in Random Engine for testing
- Training data collection for machine learning
- Move history log
- Game state management (start, reset, game over detection)

## Requirements

- Python 3.x
- pygame
- pygame-gui
- python-chess
- numpy
- torch (for ML features)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/gage-marshall/PySultan.git
cd PySultan
```

2. Install dependencies:
```bash
pip install pygame pygame-gui python-chess numpy torch
```

## Usage

Run the engine:
```bash
python main.py
```

### Controls

- **Mouse**: Click to select and move pieces
- **R**: Reset the game
- **Start Game**: Begin a new game
- **Reset Game**: Clear the board and start over

### Player Configuration

Use the dropdown menus to select:
- White Player (Human or AI)
- Black Player (Human or AI)

Enable/disable training data collection with the "Training Enabled" checkbox.

## Creating Custom Engines

Create your own AI engine by:

1. Create a new Python file in the `engines/` folder
2. Subclass `BaseEngine` from `engine_interface.py`
3. Implement the `compute_move()` method

Example:
```python
from engine_interface import BaseEngine
import chess

class MyEngine(BaseEngine):
    def __init__(self):
        super().__init__("My Engine")

    def compute_move(self, board: chess.Board) -> chess.Move:
        # Your AI logic here
        move = # calculate best move
        self.record_move(board, move)
        return move
```

The engine will automatically appear in the player selection dropdown.

## PyTorch Hydra Engine

The included **PyTorch Hydra** engine (`engines/hydra_engine.py`) demonstrates a complete neural network implementation with:

- **Convolutional Neural Network** (HydraNet) with policy and value heads
- **Alpha-Beta search** with neural network evaluation
- **Move ordering** using policy priors, MVV-LVA, and check detection
- **Self-play training** with automatic background updates
- **Position encoding** with 28 feature planes (pieces, castling rights, en passant, history)
- **Advanced reward shaping** for draws based on material balance

The engine improves over time through self-play when "Training Enabled" is checked. It uses NegaMax search with neural network guidance for tactical awareness.

## Training Data

When training is enabled, game history is saved to `training_data/<engine_name>/`:
- `.json` - Game moves and outcomes
- `.pth` - PyTorch model weights (if applicable)

## Project Structure

- `main.py` - Main game loop and UI management
- `board_ui.py` - Chess board rendering
- `engine_interface.py` - Base engine class and utilities
- `pieces/` - Chess piece images
- `engines/` - Custom AI engine implementations
  - `hydra_engine.py` - PyTorch Hydra neural network engine
- `training_data/` - Collected training data

## License

MIT License - see LICENSE file for details
