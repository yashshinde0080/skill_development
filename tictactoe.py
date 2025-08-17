def print_board(board):
    """Print the Tic Tac Toe board"""
    print("\n")
    for i in range(3):
        print(f" {board[i][0]} | {board[i][1]} | {board[i][2]} ")
        if i < 2:
            print("-----------")
    print("\n")

def check_win(board, player):
    """Check if a player has won"""
    # Check rows
    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2] == player:
            return True
    
    # Check columns
    for i in range(3):
        if board[0][i] == board[1][i] == board[2][i] == player:
            return True
    
    # Check diagonals
    if board[0][0] == board[1][1] == board[2][2] == player:
        return True
    if board[0][2] == board[1][1] == board[2][0] == player:
        return True
    
    return False

def is_board_full(board):
    """Check if the board is full"""
    for row in board:
        if ' ' in row:
            return False
    return True

def get_move():
    """Get player move"""
    while True:
        try:
            move = int(input("Enter your move (1-9): "))
            if 1 <= move <= 9:
                return move
            else:
                print("Please enter a number between 1 and 9.")
        except ValueError:
            print("Please enter a valid number.")

def play_game():
    """Main game function"""
    # Initialize the board
    board = [[' ' for _ in range(3)] for _ in range(3)]
    current_player = 'X'
    
    # Position mapping (1-9 to board coordinates)
    positions = {
        1: (0, 0), 2: (0, 1), 3: (0, 2),
        4: (1, 0), 5: (1, 1), 6: (1, 2),
        7: (2, 0), 8: (2, 1), 9: (2, 2)
    }
    
    print("Welcome to Tic Tac Toe!")
    print("Use numbers 1-9 to make your move as shown below:")
    print(" 1 | 2 | 3 ")
    print("-----------")
    print(" 4 | 5 | 6 ")
    print("-----------")
    print(" 7 | 8 | 9 ")
    
    while True:
        print_board(board)
        print(f"Player {current_player}'s turn")
        
        move = get_move()
        row, col = positions[move]
        
        # Check if position is available
        if board[row][col] == ' ':
            board[row][col] = current_player
        else:
            print("That position is already taken!")
            continue
        
        # Check for win
        if check_win(board, current_player):
            print_board(board)
            print(f"Player {current_player} wins!")
            break
        
        # Check for tie
        if is_board_full(board):
            print_board(board)
            print("It's a tie!")
            break
        
        # Switch player
        current_player = 'O' if current_player == 'X' else 'X'

if __name__ == "__main__":
    play_game()
