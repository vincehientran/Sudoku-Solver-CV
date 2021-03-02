import cv2
import numpy as np
import sys
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils

class SudokuCV():
    def __init__(self, imageName):
        self.board = np.zeros((9,9), dtype=int)
        image = cv2.imread(imageName)
        self.image = cv2.resize(image, (450,450), interpolation = cv2.INTER_AREA)
        self.extractBoard()
        contour = self.boardContour(self.image)
        cornersAndMidPoints = self.findCornersAndMidpoints(contour)
        self.cropAndWarp(cornersAndMidPoints)

        json = open('model.json', 'r')
        loadedModelJson = json.read()
        json.close()
        model = model_from_json(loadedModelJson)
        model.load_weights("model.h5")

        self.runCNN(model)

    # extract the sudoku board from the image
    def extractBoard(self):
        grayscaleImage = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        gaussianThreshold = cv2.adaptiveThreshold(grayscaleImage,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,51,10)
        self.image = gaussianThreshold

    # find the contour of the largest square
    def boardContour(self, binaryImage):
        contours, hierarchy = cv2.findContours(binaryImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        contour = contours[0]
        return contour

    # find the four corners of a contour
    # returns the coordinates for (topLeft, topRight, bottomLeft, bottomRight)
    def findCornersAndMidpoints(self, contour):
        pointSums = []
        pointDiffs = []
        for point in contour:
            pointSums.append(point[0][0] + point[0][1])
            pointDiffs.append(point[0][0] - point[0][1])
        pointSums = np.array(pointSums)
        pointDiffs = np.array(pointDiffs)

        maxSum = pointSums[0]
        idxMaxSum = 0

        minSum = pointSums[0]
        idxMinSum = 0

        maxDiff = pointDiffs[0]
        idxMaxDiff = 0

        minDiff = pointDiffs[0]
        idxMinDiff = 0

        for i in range(len(pointSums)):
            if pointSums[i] > maxSum:
                maxSum = pointSums[i]
                idxMaxSum = i
            if pointSums[i] < minSum:
                minSum = pointSums[i]
                idxMinSum = i
            if pointDiffs[i] > maxDiff:
                maxDiff = pointDiffs[i]
                idxMaxDiff = i
            if pointDiffs[i] < minDiff:
                minDiff = pointDiffs[i]
                idxMinDiff = i

        # each point is [x,y]
        topLeftPoint = contour[idxMinSum][0]
        topRightPoint = contour[idxMaxDiff][0]
        bottomLeftPoint = contour[idxMinDiff][0]
        bottomRightPoint = contour[idxMaxSum][0]

        cornerIndicies = [idxMinSum,idxMaxDiff,idxMinDiff,idxMaxSum]
        cornerIndicies = sorted(cornerIndicies)
        # side# is a list of all points on the same side followed by the corner point that it starts from
        side1 = contour[cornerIndicies[0]:cornerIndicies[1]] , cornerIndicies[0]
        side2 = contour[cornerIndicies[1]:cornerIndicies[2]] , cornerIndicies[1]
        side3 = contour[cornerIndicies[2]:cornerIndicies[3]] , cornerIndicies[2]
        side4 = np.array(list(contour[cornerIndicies[3]:]) + list(contour[:cornerIndicies[0]])) , cornerIndicies[3]

        leftSide = None
        bottomSide = None
        rightSide = None
        topSide = None

        if side1[1] == idxMinSum:
            leftSide = side1[0]
        elif side1[1] == idxMinDiff:
            bottomSide = side1[0]
        elif side1[1] == idxMaxSum:
            rightSide = side1[0]
        else:
            topSide = side1[0]

        if side2[1] == idxMinSum:
            leftSide = side2[0]
        elif side2[1] == idxMinDiff:
            bottomSide = side2[0]
        elif side2[1] == idxMaxSum:
            rightSide = side2[0]
        else:
            topSide = side2[0]

        if side3[1] == idxMinSum:
            leftSide = side3[0]
        elif side3[1] == idxMinDiff:
            bottomSide = side3[0]
        elif side3[1] == idxMaxSum:
            rightSide = side3[0]
        else:
            topSide = side3[0]

        if side4[1] == idxMinSum:
            leftSide = side4[0]
        elif side4[1] == idxMinDiff:
            bottomSide = side4[0]
        elif side4[1] == idxMaxSum:
            rightSide = side4[0]
        else:
            topSide = side4[0]

        sumLeft = [0,0]
        for point in leftSide:
            sumLeft[0] += point[0][0]
            sumLeft[1] += point[0][1]
        avgLeft = [sumLeft[0]/len(leftSide),sumLeft[1]/len(leftSide)]

        sumBottom = [0,0]
        for point in bottomSide:
            sumBottom[0] += point[0][0]
            sumBottom[1] += point[0][1]
        avgBottom = [sumBottom[0]/len(bottomSide),sumBottom[1]/len(bottomSide)]

        sumRight = [0,0]
        for point in rightSide:
            sumRight[0] += point[0][0]
            sumRight[1] += point[0][1]
        avgRight = [sumRight[0]/len(rightSide),sumRight[1]/len(rightSide)]

        sumTop = [0,0]
        for point in topSide:
            sumTop[0] += point[0][0]
            sumTop[1] += point[0][1]
        avgTop = [sumTop[0]/len(topSide),sumTop[1]/len(topSide)]

        avgLeft = np.array([int(avgLeft[0]),int(avgLeft[1])])
        avgBottom = np.array([int(avgBottom[0]),int(avgBottom[1])])
        avgRight = np.array([int(avgRight[0]),int(avgRight[1])])
        avgTop = np.array([int(avgTop[0]),int(avgTop[1])])

        return (topLeftPoint, topRightPoint, bottomLeftPoint, bottomRightPoint
        , avgLeft, avgBottom, avgRight, avgTop)


    # crop and warp the sudoku board
    def cropAndWarp(self, points):
        sides = [(points[0],points[1]), (points[1],points[2]), (points[2],points[3]), (points[3],points[0])]
        sideDistances = []
        for endpointX, endpointY in sides:
            sideDistances.append(np.linalg.norm(endpointX-endpointY))
        maxSideDistance = 450

        # after cropping and warping, the 4 corners of the board will be the 4 corners of the image
        warpedPoints = np.array([[0, 0], [maxSideDistance - 1, 0], [0, maxSideDistance - 1], [maxSideDistance - 1, maxSideDistance - 1]], dtype='float32')
        originalPoints = np.array([points[0], points[1], points[2], points[3]], dtype='float32')
        M = cv2.getPerspectiveTransform(originalPoints, warpedPoints)
        self.image = cv2.warpPerspective(self.image, M, (int(maxSideDistance), int(maxSideDistance)) )

        cv2.imshow('Cropped and Warped Image', self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        middleLeft = [0, int(maxSideDistance//2)]
        middleBottom = [int(maxSideDistance//2), int(maxSideDistance)-1]
        middleRight = [int(maxSideDistance)-1, int(maxSideDistance//2)]
        middleTop = [int(maxSideDistance//2),0]

        # move the midpoints inward towards the center of the image until we reach the outer edge of the sudoku board
        leftConcaved = False
        bottomConcaved = False
        rightConcaved = False
        topConcaved = False
        while middleLeft[0] < maxSideDistance-1 and self.image[middleLeft[1],middleLeft[0]] < 200:
            # move rightward
            middleLeft[0] += 1
        if middleLeft[0] > 30:
            leftConcaved = True

        while middleBottom[1] > 0 and self.image[middleBottom[1],middleBottom[0]] < 200:
            # move rightward
            middleBottom[1] -= 1
        if middleBottom[1] < maxSideDistance-30:
            bottomConcaved = True

        while middleRight[0] > 0 and self.image[middleRight[1],middleRight[0]] < 200:
            # move rightward
            middleRight[0] -= 1
        if middleRight[0] < maxSideDistance-30:
            rightConcaved = True

        while middleTop[1] < maxSideDistance-1 and self.image[middleTop[1],middleTop[0]] < 200:
            # move rightward
            middleTop[1] += 1
        if middleTop[1] > 30:
            topConcaved = True

        # after cropping and warping, the 4 midpoints of the board will be the 4 midpoints of the image
        # top left quadrant
        warpedPoints = None
        originalPoints = None
        if topConcaved and leftConcaved:
            warpedPoints = np.array([[0, 0], [maxSideDistance//2, 50], [50, maxSideDistance//2], [maxSideDistance//2, maxSideDistance//2]], dtype='float32')
            originalPoints = np.array([[0, 0],middleTop, middleLeft, [maxSideDistance//2, maxSideDistance//2]], dtype='float32')
        elif topConcaved and not leftConcaved:
            warpedPoints = np.array([[0, 0], [maxSideDistance//2, 50], [0, maxSideDistance//2], [maxSideDistance//2, maxSideDistance//2]], dtype='float32')
            originalPoints = np.array([[0, 0],middleTop, middleLeft, [maxSideDistance//2, maxSideDistance//2]], dtype='float32')
        elif not topConcaved and leftConcaved:
            warpedPoints = np.array([[0, 0], [maxSideDistance//2, 0], [50, maxSideDistance//2], [maxSideDistance//2, maxSideDistance//2]], dtype='float32')
            originalPoints = np.array([[0, 0],middleTop, middleLeft, [maxSideDistance//2, maxSideDistance//2]], dtype='float32')
        else:
            warpedPoints = np.array([[0, 0], [maxSideDistance//2, 0], [0, maxSideDistance//2], [maxSideDistance//2, maxSideDistance//2]], dtype='float32')
            originalPoints = np.array([[0, 0],middleTop, middleLeft, [maxSideDistance//2, maxSideDistance//2]], dtype='float32')
        M = cv2.getPerspectiveTransform(originalPoints, warpedPoints)
        quadrantTopLeft = cv2.warpPerspective(self.image, M, (int(maxSideDistance//2), int(maxSideDistance//2)) )

        # top right quadrant
        if topConcaved and rightConcaved:
            warpedPoints = np.array([[0, 50], [maxSideDistance//2, 0], [0, maxSideDistance//2], [(maxSideDistance//2)-50, maxSideDistance//2]], dtype='float32')
            originalPoints = np.array([middleTop, [maxSideDistance - 1, 0], [maxSideDistance//2, maxSideDistance//2], middleRight], dtype='float32')
        elif topConcaved and not rightConcaved:
            warpedPoints = np.array([[0, 50], [maxSideDistance//2, 0], [0, maxSideDistance//2], [maxSideDistance//2, maxSideDistance//2]], dtype='float32')
            originalPoints = np.array([middleTop, [maxSideDistance - 1, 0], [maxSideDistance//2, maxSideDistance//2], middleRight], dtype='float32')
        elif not topConcaved and rightConcaved:
            warpedPoints = np.array([[0, 0], [maxSideDistance//2, 0], [0, maxSideDistance//2], [(maxSideDistance//2)-50, maxSideDistance//2]], dtype='float32')
            originalPoints = np.array([middleTop, [maxSideDistance - 1, 0], [maxSideDistance//2, maxSideDistance//2], middleRight], dtype='float32')
        else:
            warpedPoints = np.array([[0, 0], [maxSideDistance//2, 0], [0, maxSideDistance//2], [maxSideDistance//2, maxSideDistance//2]], dtype='float32')
            originalPoints = np.array([middleTop, [maxSideDistance - 1, 0], [maxSideDistance//2, maxSideDistance//2], middleRight], dtype='float32')
        M = cv2.getPerspectiveTransform(originalPoints, warpedPoints)
        quadrantTopRight = cv2.warpPerspective(self.image, M, (int(maxSideDistance//2), int(maxSideDistance//2)) )

        # bottom left quadrant
        if bottomConcaved and leftConcaved:
            warpedPoints = np.array([[50, 0], [maxSideDistance//2, 0], [0, maxSideDistance//2], [maxSideDistance//2, (maxSideDistance//2)-50]], dtype='float32')
            originalPoints = np.array([middleLeft, [maxSideDistance//2, maxSideDistance//2], [0, maxSideDistance - 1], middleBottom], dtype='float32')
        elif bottomConcaved and not leftConcaved:
            warpedPoints = np.array([[0, 0], [maxSideDistance//2, 0], [0, maxSideDistance//2], [maxSideDistance//2, (maxSideDistance//2)-50]], dtype='float32')
            originalPoints = np.array([middleLeft, [maxSideDistance//2, maxSideDistance//2], [0, maxSideDistance - 1], middleBottom], dtype='float32')
        elif not bottomConcaved and leftConcaved:
            warpedPoints = np.array([[50, 0], [maxSideDistance//2, 0], [0, maxSideDistance//2], [maxSideDistance//2, maxSideDistance//2]], dtype='float32')
            originalPoints = np.array([middleLeft, [maxSideDistance//2, maxSideDistance//2], [0, maxSideDistance - 1], middleBottom], dtype='float32')
        else:
            warpedPoints = np.array([[0, 0], [maxSideDistance//2, 0], [0, maxSideDistance//2], [maxSideDistance//2, maxSideDistance//2]], dtype='float32')
            originalPoints = np.array([middleLeft, [maxSideDistance//2, maxSideDistance//2], [0, maxSideDistance - 1], middleBottom], dtype='float32')
        M = cv2.getPerspectiveTransform(originalPoints, warpedPoints)
        quadrantBottomLeft = cv2.warpPerspective(self.image, M, (int(maxSideDistance//2), int(maxSideDistance//2)) )

        # bottom right quadrant
        if bottomConcaved and rightConcaved:
            warpedPoints = np.array([[0, 0], [(maxSideDistance//2)-50, 0], [0, (maxSideDistance//2)-50], [maxSideDistance//2, maxSideDistance//2]], dtype='float32')
            originalPoints = np.array([[maxSideDistance//2, maxSideDistance//2], middleRight, middleBottom, [maxSideDistance - 1, maxSideDistance - 1]], dtype='float32')
        elif bottomConcaved and not rightConcaved:
            warpedPoints = np.array([[0, 0], [maxSideDistance//2, 0], [0, (maxSideDistance//2)-50], [maxSideDistance//2, maxSideDistance//2]], dtype='float32')
            originalPoints = np.array([[maxSideDistance//2, maxSideDistance//2], middleRight, middleBottom, [maxSideDistance - 1, maxSideDistance - 1]], dtype='float32')
        elif not bottomConcaved and rightConcaved:
            warpedPoints = np.array([[0, 0], [(maxSideDistance//2)-50, 0], [0, maxSideDistance//2], [maxSideDistance//2, maxSideDistance//2]], dtype='float32')
            originalPoints = np.array([[maxSideDistance//2, maxSideDistance//2], middleRight, middleBottom, [maxSideDistance - 1, maxSideDistance - 1]], dtype='float32')
        else:
            warpedPoints = np.array([[0, 0], [maxSideDistance//2, 0], [0, maxSideDistance//2], [maxSideDistance//2, maxSideDistance//2]], dtype='float32')
            originalPoints = np.array([[maxSideDistance//2, maxSideDistance//2], middleRight, middleBottom, [maxSideDistance - 1, maxSideDistance - 1]], dtype='float32')
        M = cv2.getPerspectiveTransform(originalPoints, warpedPoints)
        quadrantBottomRight = cv2.warpPerspective(self.image, M, (int(maxSideDistance//2), int(maxSideDistance//2)) )

        # combine all 4 quadrants
        topHalf = np.concatenate((quadrantTopLeft, quadrantTopRight), axis=1)
        bottomHalf = np.concatenate((quadrantBottomLeft, quadrantBottomRight), axis=1)
        self.image = np.concatenate((topHalf, bottomHalf), axis=0)

    # crop out all 81 cells and
    # run each cell image through the convolutional network
    def runCNN(self, model):
        print(self.board)

        cellSize = 450//9
        for j in range(9):

            for i in range(9):
                topLeft = (i * cellSize, j * cellSize)
                bottomRight = ((i + 1) * cellSize, (j + 1) * cellSize)
                cellImage = self.image[topLeft[0]:bottomRight[0], topLeft[1]:bottomRight[1]]
                numberImageZoom = self.image[topLeft[0]+10:bottomRight[0]-10, topLeft[1]+10:bottomRight[1]-10]
                numberImage = self.image[topLeft[0]+5:bottomRight[0]-5, topLeft[1]+5:bottomRight[1]-5]

                # count white pixels
                sumOfWhitePixels = 0
                for x in range(len(numberImageZoom[0])):
                    for y in range(len(numberImageZoom)):
                        if numberImageZoom[x][y] > 127:
                            sumOfWhitePixels += 1

                # if there are more than 10% white pixel
                if sumOfWhitePixels > ((cellSize-20)**2)*0.05:
                    #modelPrediction = model.predict_classes(numberImage,verbose=0)
                    #print(modelPrediction[0])
                    cv2.imshow('Cropped and Warped Image', numberImage)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()


def main():
    if len(sys.argv) != 2:
        return
    else:
        sudoku = SudokuCV(sys.argv[1])

if __name__ == '__main__':
    main()
