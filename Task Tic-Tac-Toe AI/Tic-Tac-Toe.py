import math

# Initialize the board
board = [" " for _ in range(9)]

# Function to print the board
def print_board():
    print()
    for row in [board[i*3:(i+1)*3] for i in range(3)]:
        print("| " + " | ".join(row) + " |")
    print()

# Check if a player has won
def check_winner(brd, player):
    win_cond = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # cols
        [0, 4, 8], [2, 4, 6]              # diagonals
    ]
    for cond in win_cond:
        if all(brd[i] == player for i in cond):
            return True
    return False

# Check if board is full
def is_full(brd):
    return " " not in brd

# Minimax algorithm
def minimax(brd, depth, is_maximizing):
    if check_winner(brd, "O"):  # AI win
        return 1
    if check_winner(brd, "X"):  # Human win
        return -1
    if is_full(brd):
        return 0

    if is_maximizing:  # AI's turn
        best_score = -math.inf
        for i in range(9):
            if brd[i] == " ":
                brd[i] = "O"
                score = minimax(brd, depth + 1, False)
                brd[i] = " "
                best_score = max(score, best_score)
        return best_score
    else:  # Human's turn
        best_score = math.inf
        for i in range(9):
            if brd[i] == " ":
                brd[i] = "X"
                score = minimax(brd, depth + 1, True)
                brd[i] = " "
                best_score = min(score, best_score)
        return best_score

# AI Move
def ai_move():
    best_score = -math.inf
    move = 0
    for i in range(9):
        if board[i] == " ":
            board[i] = "O"
            score = minimax(board, 0, False)
            board[i] = " "
            if score > best_score:
                best_score = score
                move = i
    board[move] = "O"

# Main game loop
def play_game():
    print("Tic-Tac-Toe: You are X, AI is O")
    print_board()

    while True:
        # Human move
        move = int(input("Enter your move (1-9): ")) - 1
        if board[move] != " ":
            print("Invalid move, try again!")
            continue
        board[move] = "X"

        print_board()

        if check_winner(board, "X"):
            print("🎉 You win!")
            break
        elif is_full(board):
            print("It's a draw!")
            break

        # AI move
        ai_move()
        print("AI has made its move:")
        print_board()

        if check_winner(board, "O"):
            print("🤖 AI wins!")
            break
        elif is_full(board):
            print("It's a draw!")
            break

play_game()
