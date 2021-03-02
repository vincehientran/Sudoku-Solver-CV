class Solution(object):
    def solveSudoku(self, board):
        """
        :type board: List[List[str]]
        :rtype: None Do not return anything, modify board in-place instead.
        """
        self.board = board
        self.solve()

    def solve(self):
        row, col = self.findEmpty()
        if row == -1 and col == -1:
            return True
        values = ['1','2','3','4','5','6','7','8','9']
        for val in values:
            if self.isValid(row, col, val):
                self.board[row][col] = val
                if self.solve():
                    return True
                # val does at row and col does not
                # lead to the correct answer
                # undo and try next val
                self.board[row][col] = '.'
        return False

    def findEmpty(self):
        for row in range(len(self.board)):
            for col in range(len(self.board[0])):
                if self.board[row][col] == '.':
                    return row, col

        # no empty cell found
        return -1, -1

    def isValid(self, row, col, val):
        if self.checkRow(row, val) and self.checkCol(col, val) and self.checkBox(row, col, val):
            return True
        return False


    def checkRow(self, row, val):
        for col in range(len(self.board[row])):
            if val == self.board[row][col]:
                return False
        return True

    def checkCol(self, col, val):
        for row in range(len(self.board)):
            if val == self.board[row][col]:
                return False
        return True

    def checkBox(self, row, col, val):
        startRow = row - (row % 3)
        startCol = col - (col % 3)
        for r in range(startRow, startRow + 3):
            for c in range(startCol, startCol + 3):
                if val == self.board[r][c]:
                    return False
        return True
