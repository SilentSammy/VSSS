import sys
from pathlib import Path
import os

# Add parent directory to path to import board_config
sys.path.insert(0, str(Path(__file__).parent.parent))

from board_config import board_config, image_to_pdf

# Set script's directory as current working directory
script_dir = Path(__file__).parent
os.chdir(script_dir)

# Check if board.png exists
board_image_path = script_dir / "board.png"

if not board_image_path.exists():
    # Generate the board image
    print("Generating board.png...")
    board_config.save_image(str(board_image_path), width_px=2400, margin_m=0.01)
    print(f"Board image created at {board_image_path}")
    print("You can now modify the image if needed, then run this script again to generate the PDF.")
else:
    # Generate PDF from existing image
    print("Found board.png, generating PDF...")
    import cv2
    img = cv2.imread(str(board_image_path))
    pdf_path = script_dir / "board.pdf"
    image_to_pdf(img, str(pdf_path), board_config.board_width)
    print(f"PDF generated at {pdf_path}")

