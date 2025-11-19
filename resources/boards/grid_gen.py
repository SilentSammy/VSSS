import cv2
import os

def main():
    # 1) Load your dictionary and build the board
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    
    # 2) Calculate marker dimensions for 60×90 cm board with 3×3 markers
    # Total board dimensions in meters
    board_width = 0.60
    
    # 3×3 grid: calculate marker size and separation
    # We'll use markers with spacing to fill the board evenly
    marker_length = 0.075
    # Use consistent separation for both width and height to keep markers square
    # Calculate based on width (60 cm - 3*7.5 cm = 37.5 cm / 2 gaps = 18.75 cm each)
    marker_separation = (board_width - 3 * marker_length) / 2
    
    board = cv2.aruco.GridBoard(
        size=(3, 4),
        markerLength=marker_length,      # marker side length in meters
        markerSeparation=marker_separation,  # gap between markers in meters
        dictionary=aruco_dict
    )

    # 2) Draw the board into an image - calculate dimensions based on actual board size
    # Calculate actual board dimensions including markers and separations
    markers_x, markers_y = board.getGridSize()
    actual_board_width = markers_x * marker_length + (markers_x - 1) * marker_separation
    actual_board_height = markers_y * marker_length + (markers_y - 1) * marker_separation
    
    # Calculate aspect ratio and image dimensions
    aspect_ratio = actual_board_height / actual_board_width
    img_width = 1200  # base width in pixels
    img_height = int(img_width * aspect_ratio)
    
    board_img = board.generateImage(
        (img_width, img_height),  # size of the output image
        marginSize=0          # margin around the board in pixels
    )

    # 3) Save alongside this script
    out_file = os.path.join(os.path.dirname(__file__), "gridboard.png")
    cv2.imwrite(out_file, board_img)
    print(f"Saved board image to: {out_file}")

if __name__ == "__main__":
    main()
