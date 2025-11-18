#!/usr/bin/env python3
"""
Render a 2×2 ArUco GridBoard to a PNG image.
"""

import cv2
import os

def main():
    # 1) Load your dictionary and build the board
    board = cv2.aruco.CharucoBoard(
        size=(9, 24),
        squareLength=0.1,
        markerLength=0.08,
        dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    )

    # 2) Draw the board into an image
    img_size = 900  # base size for the output

    # Option to keep the image square or fit the board snugly
    keep_square = False  # Set to True for square image, False for snug fit

    if keep_square:
        # Square image: both dimensions are img_size
        out_size = (img_size, img_size)
        margin = 0
    else:
        # Fit the board snugly: calculate based on board aspect ratio
        squares_x, squares_y = board.getChessboardSize()
        aspect_ratio = squares_y / squares_x
        width = img_size
        height = int(img_size * aspect_ratio)
        out_size = (width, height)
        margin = 0

    board_img = board.generateImage(
        out_size,  # size of the output image
        marginSize=margin
    )

    # 3) Save alongside this script
    out_file = os.path.join(os.path.dirname(__file__), "gridboard.png")
    cv2.imwrite(out_file, board_img)
    print(f"Saved board image to: {out_file}")

if __name__ == "__main__":
    main()
