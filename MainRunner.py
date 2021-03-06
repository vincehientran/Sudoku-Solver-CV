import sys
from SudokuCV import SudokuCV
from SudokuSolver import Solution
import numpy as np

def main():
    if len(sys.argv) != 2:
        return
    else:
        sudoku = SudokuCV(sys.argv[1])
        board = sudoku.getBoard()
        solution = Solution()
        solution.solveSudoku(board)
        print('\nSolution')
        print(np.array(board).astype(np.int8))

if __name__ == '__main__':
    main()
