import os
import time

def print_board(board):
    """Prints the Tic-Tac-Toe board."""
    os.system('clear' if os.name == 'posix' else 'cls')
    print("   1   2   3")
    print("  -----------")
    for i, row in enumerate(board):
        print(f"{i + 1} | {' | '.join(row)}|")
        print("  -----------")

def check_win(board, player):
    """Checks if a player has won."""
    # Check rows
    for row in board:
        if all(s == player for s in row):
            return True
    # Check columns
    for col in range(3):
        if all(board[row][col] == player for row in range(3)):
            return True
    # Check diagonals
    if all(board[i][i] == player for i in range(3)) or \
       all(board[i][2 - i] == player for i in range(3)):
        return True
    return False

def check_draw(board):
    """Checks if the game is a draw."""
    return all(cell != " " for row in board for cell in row)

def main():
    """Main function to run the Tic-Tac-Toe game."""
    scores = {"X": 0, "O": 0}
    paused = False

    while True:
        board = [[" " for _ in range(3)] for _ in range(3)]
        current_player = "X"
        game_over = False

        while not game_over:
            if paused:
                print_board(board)
                print(f"Scores: X - {scores['X']}, O - {scores['O']}")
                cmd = input("Game is paused. Enter 'resume' to continue or 'end' to quit: ").lower()
                if cmd == 'resume':
                    paused = False
                elif cmd == 'end':
                    print("Thanks for playing! Final Scores:")
                    print(f"X - {scores['X']}, O - {scores['O']}")
                    return
                continue

            print_board(board)
            print(f"Scores: X - {scores['X']}, O - {scores['O']}")
            move = input(f"Player {current_player}, enter move (r <row> c <col>), or type 'pause', 'stop', or 'end': ").lower()

            if move in ['pause', 'stop']:
                paused = True
                continue
            elif move == 'end':
                print("Thanks for playing! Final Scores:")
                print(f"X - {scores['X']}, O - {scores['O']}")
                return

            try:
                parts = move.split()
                if len(parts) == 4 and parts[0] == 'r' and parts[2] == 'c':
                    row, col = int(parts[1]) - 1, int(parts[3]) - 1
                    if 0 <= row <= 2 and 0 <= col <= 2 and board[row][col] == " ":
                        board[row][col] = current_player
                        if check_win(board, current_player):
                            print_board(board)
                            print(f"Player {current_player} wins this round!")
                            scores[current_player] += 1
                            game_over = True
                        elif check_draw(board):
                            print_board(board)
                            print("This round is a draw!")
                            game_over = True
                        else:
                            current_player = "O" if current_player == "X" else "X"
                    else:
                        print("Invalid move. Cell might be taken or out of bounds. Try again.")
                        time.sleep(1.5)
                else:
                    raise ValueError
            except (ValueError, IndexError):
                print("Invalid input. Please use 'r <row> c <col>' format (e.g., 'r 1 c 2').")
                time.sleep(1.5)
        
        print("Starting new round in 3 seconds...")
        time.sleep(3)


if __name__ == "__main__":
    main()
