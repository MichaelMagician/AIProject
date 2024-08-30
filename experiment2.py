import pygame
import random

pygame.init()

# Constants
SCREEN_WIDTH = 300
SCREEN_HEIGHT = 500
GRID_SIZE = 30
GRID_WIDTH = SCREEN_WIDTH // GRID_SIZE
GRID_HEIGHT = SCREEN_HEIGHT // GRID_SIZE

# Colors
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
CYAN = (0, 255, 255)
ORANGE = (255, 165, 0)
YELLOW = (255, 255, 0)
PURPLE = (128, 0, 128)
GREEN = (0, 128, 0)
RED = (255, 0, 0)
BLACK = (0, 0, 0)
colors =[WHITE,BLUE,CYAN,ORANGE,YELLOW,PURPLE,GREEN,RED,BLACK]

# Define each type of tetromino (piece)
tetrominos = [
    [[1, 1, 1, 1]],             # I-shape
    [[1, 1, 1], [0, 1, 0]],     # T-shape
    [[1, 1, 1], [1, 0, 0]],     # L-shape
    [[1, 1, 1], [0, 0, 1]],     # J-shape
    [[1, 1], [1, 1]],           # O-shape
    [[0, 1, 1], [1, 1, 0]],     # Z-shape
    [[1, 1, 0], [0, 1, 1]]      # S-shape
]

# Initialize the game window
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Tetris")

# Clock to control the game speed
clock = pygame.time.Clock()


class Piece:
    def __init__(self, shape, color):
        self.shape = shape
        self.color = color
        self.rotation = 0
        self.x = GRID_WIDTH // 2 - len(shape[0]) // 2
        self.y = 0

    def rotate(self):
        self.rotation = (self.rotation + 1) % 4
        self.shape = tetrominos[self.type][self.rotation]

    def move_down(self):
        self.y += 1

    def move_sideways(self, dx):
        self.x += dx

    def draw(self):
        for row in range(len(self.shape)):
            for col in range(len(self.shape[0])):
                if self.shape[row][col] == 1:
                    pygame.draw.rect(screen, self.color, 
                                     pygame.Rect((self.x + col) * GRID_SIZE, 
                                                 (self.y + row) * GRID_SIZE, 
                                                 GRID_SIZE, GRID_SIZE))
def check_collision(board, piece):
    for row in range(len(piece.shape)):
        for col in range(len(piece.shape[0])):
            if piece.shape[row][col] == 1:
                if (piece.y + row >= GRID_HEIGHT or 
                    piece.x + col < 0 or 
                    piece.x + col >= GRID_WIDTH or 
                    board[piece.y + row][piece.x + col] != BLACK):
                    return True
    return False

def merge_piece(board, piece):
    for row in range(len(piece.shape)):
        for col in range(len(piece.shape[0])):
            if piece.shape[row][col] == 1:
                if piece.y + row < GRID_HEIGHT and piece.x + col < GRID_WIDTH:    
                    board[piece.y + row][piece.x + col] = piece.color

def draw_board(board):
    for row in range(GRID_HEIGHT):
        for col in range(GRID_WIDTH):
            pygame.draw.rect(screen, board[row][col], 
                             pygame.Rect(col * GRID_SIZE, row * GRID_SIZE, 
                                         GRID_SIZE, GRID_SIZE))

def remove_full_rows(board):
    rows_to_remove = []
    for row in range(GRID_HEIGHT):
        if all(cell != BLACK for cell in board[row]):
            rows_to_remove.append(row)
    for row in rows_to_remove:
        del board[row]
        board.insert(0, [BLACK] * GRID_WIDTH)

def main():
    board = [[BLACK for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
    piece = Piece(random.choice(tetrominos), random.choice(colors))

    while True:
        screen.fill(BLACK)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    piece.move_sideways(-1)
                    if check_collision(board, piece):
                        piece.move_sideways(1)
                elif event.key == pygame.K_RIGHT:
                    piece.move_sideways(1)
                    if check_collision(board, piece):
                        piece.move_sideways(-1)
                elif event.key == pygame.K_DOWN:
                    piece.move_down()
                    if check_collision(board, piece):
                        piece.move_down()
                        # merge_piece(board, piece)
                        remove_full_rows(board)
                        piece = Piece(random.choice(tetrominos), random.choice(colors))
                elif event.key == pygame.K_UP:
                    piece.rotate()
                    if check_collision(board, piece):
                        piece.rotate()
        
            piece.move_down()
            if check_collision(board, piece):
                # piece.move_down()
                merge_piece(board, piece)
                remove_full_rows(board)
                piece = Piece(random.choice(tetrominos), random.choice(colors))
            else:
                piece.move_down()
            
        draw_board(board)
        piece.draw()
        pygame.display.flip()
        clock.tick(10)

if __name__ == "__main__":
    main()
                    