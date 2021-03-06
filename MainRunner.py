import sys
from SudokuCV import SudokuCV
from SudokuSolver import Solution
import numpy as np

def main():
    if len(sys.argv) != 2:
        return
    else:
        sudoku = SudokuCV(sys.argv[1])
        solution = Solution()
        solution.solveSudoku(sudoku)

        print('\nSolution')
        print(np.array(solution.getBoard()).astype(np.int8))

if __name__ == '__main__':
    main()
