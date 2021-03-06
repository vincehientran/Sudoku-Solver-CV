import cv2
import numpy as np
from SudokuCV import SudokuCV

class Solution(object):
    def solveSudoku(self, sudoku):
        """
        :type board: List[List[str]]
        :rtype: None Do not return anything, modify board in-place instead.
        """
        self.board = sudoku.getBoard()

        # keeps track of which cell has the original numbers on the sudoku board
        # empty cells are True
        orginalNumbers = []
        for row in self.board:
            tempRow = []
            for val in row:
                if val == '.':
                    tempRow.append(True)
                else:
                    tempRow.append(False)
            orginalNumbers.append(tempRow)

        self.solve()

        cellSize = 450//9
        transparent_img = np.zeros((450, 450, 3), dtype=np.uint8)

        for i in range(9):
            for j in range(9):

                if orginalNumbers[j][i]:
                    textLocation = (i * cellSize +15, (j + 1) * cellSize -15)
                    cv2.putText(transparent_img,str(self.board[j][i]),textLocation, cv2.FONT_HERSHEY_SIMPLEX, 1, (20, 20, 255, 0), 1, cv2.LINE_AA)

        originalImage, cornersAndMidPoints = sudoku.getImageAndContours()
        originalSize = originalImage.shape
        x, y, _ = originalSize

        maxSideDistance = 450
        originalPoints = np.array([[0, 0], [maxSideDistance - 1, 0], [0, maxSideDistance - 1], [maxSideDistance - 1, maxSideDistance - 1]], dtype='float32')
        warpedPoints = np.array([cornersAndMidPoints[0], cornersAndMidPoints[1], cornersAndMidPoints[2], cornersAndMidPoints[3]], dtype='float32')
        M = cv2.getPerspectiveTransform(originalPoints, warpedPoints)
        transparent_img = cv2.warpPerspective(transparent_img, M, (y,x) )
        tmp = cv2.cvtColor(transparent_img, cv2.COLOR_BGR2GRAY)
        _,alpha = cv2.threshold(tmp,19,255,cv2.THRESH_BINARY)
        b, g, r = cv2.split(transparent_img)
        rgba = [b,g,r, alpha]
        dst = cv2.merge(rgba,4)
        cv2.imwrite("test.png", dst)

        # Extract the alpha mask of the RGBA image, convert to RGB
        b,g,r,a = cv2.split(dst)
        overlay_color = cv2.merge((b,g,r))

        mask = cv2.medianBlur(a,1)

        # Black-out the area behind the logo in our original ROI
        img1_bg = cv2.bitwise_and(originalImage.copy(),originalImage.copy(),mask = cv2.bitwise_not(mask))

        # Mask out the logo from the logo image.
        img2_fg = cv2.bitwise_and(overlay_color,overlay_color,mask = mask)

        # Update the original image with our new ROI
        bg_img = cv2.add(img1_bg, img2_fg)


        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGRA2BGR)
        height, width, _ = bg_img.shape
        ratio = height/width
        cv2.imwrite('solution.jpg', bg_img)
        bg_img = cv2.resize(bg_img, (int(900//ratio),900), interpolation = cv2.INTER_AREA)
        cv2.imshow('solution.jpg', bg_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

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

    def getBoard(self):
        return self.board
