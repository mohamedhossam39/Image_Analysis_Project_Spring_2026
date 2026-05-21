import glob
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

class Project_MS_1:
    """
    Main project class designed to cover Computer Vision topics to parse and solve a 
    Sudoku puzzle grid from a real-life captured image.
    """

    def __init__(self, image_paths):
        """
        Initializes the class by loading a list of image paths into a list of image arrays.
        """
        self.images = [
            cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths
        ]

        for idx, img in enumerate(self.images):
            if img is None:
                print(f"Warning: Failed to load image at index {idx}. Check the file path.")

        #Function call to Build template bank for OCR implementation without machine learning
        self.template_bank = self._build_template_bank()
        self.show_images_grid()

    def show_images_grid(self):
        """Display the initial input images grid using Matplotlib."""
        try:
            plt.figure(figsize=(12, 12))
            for idx, img in enumerate(self.images):
                if img is not None:
                    plt.subplot(4, 4, idx + 1)
                    plt.imshow(img, cmap="gray")
                    plt.title(f"Image {idx + 1}")
                    plt.axis("off")
            plt.tight_layout()
            print("\n[INFO] Displaying the initial input images grid.")
            print("[INFO] CLOSE the plot window to begin the processing pipeline...")
            plt.show(block=True)
        except Exception as e:
            print(f"Error showing image grid: {e}")
        finally:
            self.pre_process_images()

    def pre_process_images(self):
        """
        Applies robust image preprocessing, enhancement, and noise attenuation.
        Fulfills Milestone 1 requirement for Preprocessing of the captured image.
        """
        try:
            self.processed_images = []
            
            # Using CLAHE for reliable image enhancement 
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

            for idx, img in enumerate(self.images):
                if img is None:
                    self.processed_images.append(None)
                    continue

                enhanced = clahe.apply(img)
                
                # Noise attenuation via Gaussian Blurring
                blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
                
                # Binarization to obtain a clean grid for contour detection and OCR
                thresh = cv2.adaptiveThreshold(
                    blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 45, 5
                )
                
                kernel = np.ones((3, 3), np.uint8)
                processed = cv2.dilate(thresh, kernel, iterations=1)

                self.processed_images.append(processed)

        except Exception as e:
            print(f"Error in preprocessing: {e}")
        finally:
            self.isolate_outer_frame()

    def isolate_outer_frame(self):
        """
        Extracts puzzle information via outer frame isolation and corners identification.
        """
        try:
            self.framed_images = []
            self.corners_list = []

            for idx, img in enumerate(self.processed_images):
                if img is None:
                    self.corners_list.append(None)
                    self.framed_images.append(None)
                    continue

                img_area = img.shape[0] * img.shape[1]
                
                # Outer frame isolation 
                contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                display_img = cv2.cvtColor(self.images[idx], cv2.COLOR_GRAY2BGR)
                grid_corners = None

                if contours:
                    contours = sorted(contours, key=cv2.contourArea, reverse=True)
                    for cnt in contours:
                        area = cv2.contourArea(cnt)
                        if area < img_area * 0.15 or area > img_area * 0.90:
                            continue
                            
                        perimeter = cv2.arcLength(cnt, True)
                        
                        # Outer frame corners identification 
                        for eps in np.linspace(0.01, 0.08, 15):
                            approx = cv2.approxPolyDP(cnt, eps * perimeter, True)
                            if len(approx) == 4 and cv2.isContourConvex(approx):
                                grid_corners = approx
                                break
                        if grid_corners is not None:
                            break

                self.corners_list.append(grid_corners)
                self.framed_images.append(display_img)

        except Exception as e:
            print(f"Error isolating frame: {e}")
        finally:
            self.straighten_grids()

    def straighten_grids(self):
        """
        Applies required geometric transformations to straighten the puzzle for OCR.
        Fulfills Milestone 1 requirement for Grid straightening into a square.
        """
        try:
            self.straightened_images = []
            side = 450

            for idx, corners in enumerate(self.corners_list):
                original_img = self.images[idx]
                if original_img is None:
                    self.straightened_images.append(None)
                    continue

                if corners is not None:
                    # Applying geometric transformations to straighten the grid
                    pts = corners.reshape(4, 2)
                    rect = np.zeros((4, 2), dtype="float32")
                    s = pts.sum(axis=1)
                    rect[0] = pts[np.argmin(s)]
                    rect[2] = pts[np.argmax(s)]
                    diff = np.diff(pts, axis=1)
                    rect[1] = pts[np.argmin(diff)]
                    rect[3] = pts[np.argmax(diff)]

                    dst = np.array([
                        [0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]
                    ], dtype="float32")
                    
                    M = cv2.getPerspectiveTransform(rect, dst)
                    straightened = cv2.warpPerspective(original_img, M, (side, side))
                else:
                    straightened = cv2.resize(original_img, (side, side))

                self.straightened_images.append(straightened)

        except Exception as e:
            print(f"Error straightening grids: {e}")
        finally:
            self.extract_and_recognize_puzzles()

    # ================================================================== #
    #            STRICT FALLBACK OCR ENGINE                              #
    # ================================================================== #

    def _build_template_bank(self):
        """
        Builds a template bank for manual implementation of OCR without machine learning algorithms.
        """
        bank = {d: [] for d in range(1, 10)}
        fonts = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, 
                 cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_HERSHEY_COMPLEX]

        for digit in range(1, 10):
            for font in fonts:
                for scale in [0.8, 1.0, 1.2]:
                    for th in [2, 3]:
                        temp_canvas = np.zeros((60, 60), dtype=np.uint8)
                        cv2.putText(temp_canvas, str(digit), (15, 45), font, scale, 255, th)
                        coords = cv2.findNonZero(temp_canvas)
                        if coords is not None:
                            x, y, w, h = cv2.boundingRect(coords)
                            bank[digit].append(self._normalize_digit(temp_canvas[y:y+h, x:x+w]))

        for th in [2, 3, 4]:
            temp_canvas = np.zeros((50, 50), dtype=np.uint8)
            cv2.line(temp_canvas, (25, 10), (25, 40), 255, th)
            coords = cv2.findNonZero(temp_canvas)
            if coords is not None:
                x, y, w, h = cv2.boundingRect(coords)
                bank[1].append(self._normalize_digit(temp_canvas[y:y+h, x:x+w]))
                
        return bank

    def _normalize_digit(self, roi):
        """Helper to ensure extracted digits are scale-neutral."""
        h, w = roi.shape
        if h == 0 or w == 0:
            return np.zeros((28, 28), dtype=np.uint8)

        scale = 20.0 / max(h, w)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        
        resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)

        canvas = np.zeros((28, 28), dtype=np.uint8)
        y_off, x_off = (28 - new_h) // 2, (28 - new_w) // 2
        canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
        _, canvas = cv2.threshold(canvas, 127, 255, cv2.THRESH_BINARY)
        
        canvas = cv2.GaussianBlur(canvas, (3, 3), 0)
        return canvas

    def extract_and_recognize_matrix(self, ocr_ready_img):
        """
        Parses cell values utilizing basic OCR with pattern matching.
        """
        detected_matrix = np.zeros((9, 9), dtype=int)
        cell_size = 50
        predictions = []

        for r in range(9):
            for c in range(9):
                y1, y2 = r * cell_size, (r + 1) * cell_size
                x1, x2 = c * cell_size, (c + 1) * cell_size
                cell = ocr_ready_img[y1:y2, x1:x2].copy()

                margin = 3
                cell[:margin, :] = 0
                cell[-margin:, :] = 0
                cell[:, :margin] = 0
                cell[:, -margin:] = 0

                if cv2.countNonZero(cell) < 30:
                    continue

                contours, _ = cv2.findContours(cell, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                best_cnt = None
                max_area = 0
                
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    x, y, w, h = cv2.boundingRect(cnt)
                    
                    if area < 20 or area > 800 or w < 3 or h < 10:
                        continue
                        
                    cx = x + w / 2
                    cy = y + h / 2
                    if abs(cx - 25) > 15 or abs(cy - 25) > 15:
                        continue 
                        
                    if area > max_area:
                        max_area = area
                        best_cnt = cnt
                        
                if best_cnt is None:
                    continue 
                
                x, y, w, h = cv2.boundingRect(best_cnt)
                digit_roi = cell[y:y+h, x:x+w]
                aspect_ratio = w / float(h)
                
                if aspect_ratio > 1.2: 
                    continue 
                
                target_canvas = self._normalize_digit(digit_roi)
                best_digit = 0
                max_score = -1.0

                for digit, templates in self.template_bank.items():
                    if digit == 1 and aspect_ratio > 0.65: continue
                    if digit != 1 and aspect_ratio < 0.20: continue

                    for tmpl in templates:
                        # Basic OCR via Pattern Matching 
                        res = cv2.matchTemplate(target_canvas, tmpl, cv2.TM_CCOEFF_NORMED)
                        score = np.max(res)
                        if score > max_score:
                            max_score = score
                            best_digit = digit

                if best_digit in [1, 4, 7]:
                    req_score = 0.65 
                else:
                    req_score = 0.55

                if max_score > req_score:
                    predictions.append((max_score, best_digit, r, c))

        predictions.sort(key=lambda x: x[0], reverse=True)

        for score, digit, r, c in predictions:
            if np.any(detected_matrix[r, :] == digit): continue
            if np.any(detected_matrix[:, c] == digit): continue
            box_r, box_c = (r // 3) * 3, (c // 3) * 3
            if np.any(detected_matrix[box_r:box_r+3, box_c:box_c+3] == digit): continue
                
            detected_matrix[r][c] = digit

        return detected_matrix.tolist()

    def extract_and_recognize_puzzles(self):
        """Iterates over straightened grids to extract OCR digits."""
        try:
            self.detected_matrices = []

            for idx, straightened in enumerate(self.straightened_images):
                if straightened is None:
                    self.detected_matrices.append(None)
                    continue

                denoised = cv2.GaussianBlur(straightened, (5, 5), 0)
                ocr_ready = cv2.adaptiveThreshold(
                    denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 45, 5
                )

                matrix = self.extract_and_recognize_matrix(ocr_ready)
                self.detected_matrices.append(matrix)

        except Exception as e:
            print(f"Error during character recognition: {e}")
        finally:
            self.solve_puzzles()

    # ================================================================== #
    #                     SUDOKU LOGIC & DISPLAY PIPELINE                #
    # ================================================================== #

    def solve_puzzles(self):
        """
        Handles solving the extracted sudoku puzzle.
        """
        try:
            self.solved_matrices = []
            for idx, matrix in enumerate(self.detected_matrices):
                if matrix is None:
                    self.solved_matrices.append(None)
                    continue

                solved_grid = [row[:] for row in matrix]

                if not self._is_valid_starting_board(solved_grid):
                    print(f"[!] Unsolvable configuration (Initial OCR conflicts) parsed for Grid #{idx + 1}.")
                    self.solved_matrices.append(None)
                    continue

                if self.solve_sudoku(solved_grid):
                    self.solved_matrices.append(solved_grid)
                else:
                    print(f"[!] Unsolvable configuration parsed for Grid #{idx + 1}.")
                    self.solved_matrices.append(None)

        except Exception as e:
            print(f"Error during puzzle logic solving: {e}")
        finally:
            self.save_output_images()
            self.display_final_results()

    def _is_valid_starting_board(self, grid):
        """Helper function to validate the OCR parsed grid prior to solving."""
        for r in range(9):
            nums = [grid[r][c] for c in range(9) if grid[r][c] != 0]
            if len(nums) != len(set(nums)): return False
        for c in range(9):
            nums = [grid[r][c] for r in range(9) if grid[r][c] != 0]
            if len(nums) != len(set(nums)): return False
        for box_r in range(3):
            for box_c in range(3):
                nums = []
                for r in range(box_r * 3, box_r * 3 + 3):
                    for c in range(box_c * 3, box_c * 3 + 3):
                        if grid[r][c] != 0: nums.append(grid[r][c])
                if len(nums) != len(set(nums)): return False
        return True

    def solve_sudoku(self, grid):
        """
        Sudoku solution algorithm implementation. 
        Note: The project specifications state we are allowed to look up and integrate a solution algorithm.
        """
        empty_loc = self._find_mrv_empty_location(grid)
        if not empty_loc: return True

        row, col, candidates = empty_loc
        if not candidates: return False

        for num in candidates:
            grid[row][col] = num
            if self.solve_sudoku(grid): return True
            grid[row][col] = 0
        return False

    def _find_mrv_empty_location(self, grid):
        """Finds the most restricted variable (empty cell) for backtracking efficiency."""
        min_candidates = 10
        best_loc = None
        best_candidates = []

        for r in range(9):
            for c in range(9):
                if grid[r][c] == 0:
                    row_used = set(grid[r])
                    col_used = {grid[i][c] for i in range(9)}
                    start_r, start_c = r - r % 3, c - c % 3
                    box_used = {grid[start_r + br][start_c + bc] for br in range(3) for bc in range(3)}

                    used = row_used | col_used | box_used
                    candidates = [n for n in range(1, 10) if n not in used]

                    if not candidates: return (r, c, [])

                    if len(candidates) < min_candidates:
                        min_candidates = len(candidates)
                        best_loc = (r, c)
                        best_candidates = candidates
                        if min_candidates == 1: return (r, c, best_candidates)
        if best_loc:
            return (*best_loc, best_candidates)
        return None

    def save_output_images(self):
        """
        Saves output to local directory. Helps in producing working code and demo material.
        """
        print("\n[INFO] Saving output images...")
        out_dir = "Project_Outputs"
        straightened_dir = os.path.join(out_dir, "Straightened")
        solved_dir = os.path.join(out_dir, "Solved")

        os.makedirs(straightened_dir, exist_ok=True)
        os.makedirs(solved_dir, exist_ok=True)

        for idx, straightened in enumerate(self.straightened_images):
            if straightened is None: continue
            
            # Save the clean 450x450 straightened grid
            straight_path = os.path.join(straightened_dir, f"grid_{idx + 1:02d}.jpg")
            cv2.imwrite(straight_path, straightened)

            # Re-create and save the solved grid visualization
            rendered_grid = np.ones((450, 450, 3), dtype=np.uint8) * 255
            cell_size = 50

            for i in range(10):
                thick = 3 if i % 3 == 0 else 1
                cv2.line(rendered_grid, (0, i * cell_size), (450, i * cell_size), (0, 0, 0), thick)
                cv2.line(rendered_grid, (i * cell_size, 0), (i * cell_size, 450), (0, 0, 0), thick)

            original_grid = self.detected_matrices[idx]
            solved_grid = self.solved_matrices[idx]

            if original_grid:
                for r in range(9):
                    for c in range(9):
                        orig_val = original_grid[r][c]
                        if orig_val != 0:
                            cv2.putText(rendered_grid, str(orig_val), (c * cell_size + 15, r * cell_size + 35),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 180), 2)
                        elif solved_grid and solved_grid[r][c] != 0:
                            cv2.putText(rendered_grid, str(solved_grid[r][c]), (c * cell_size + 15, r * cell_size + 35),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 150, 0), 2)

            solved_path = os.path.join(solved_dir, f"grid_{idx + 1:02d}.jpg")
            cv2.imwrite(solved_path, rendered_grid)
            
        print(f"[INFO] Images successfully saved to folder: '{out_dir}'")

    def display_final_results(self):
        """
        Displays final results using Matplotlib to support the documentation, demos, and discussion phase.
        """
        try:
            for idx, straightened in enumerate(self.straightened_images):
                if straightened is None: continue

                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                axes[0].imshow(straightened, cmap="gray")
                axes[0].set_title(f"Perspective Straightened Grid #{idx + 1}")
                axes[0].axis("off")

                rendered_grid = np.ones((450, 450, 3), dtype=np.uint8) * 255
                cell_size = 50

                for i in range(10):
                    thick = 3 if i % 3 == 0 else 1
                    cv2.line(rendered_grid, (0, i * cell_size), (450, i * cell_size), (0, 0, 0), thick)
                    cv2.line(rendered_grid, (i * cell_size, 0), (i * cell_size, 450), (0, 0, 0), thick)

                original_grid = self.detected_matrices[idx]
                solved_grid = self.solved_matrices[idx]

                if original_grid:
                    for r in range(9):
                        for c in range(9):
                            orig_val = original_grid[r][c]
                            if orig_val != 0:
                                cv2.putText(rendered_grid, str(orig_val), (c * cell_size + 15, r * cell_size + 35),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 180), 2)
                            elif solved_grid and solved_grid[r][c] != 0:
                                cv2.putText(rendered_grid, str(solved_grid[r][c]), (c * cell_size + 15, r * cell_size + 35),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 150, 0), 2)

                axes[1].imshow(rendered_grid)
                axes[1].set_title(f"Computed Matrix #{idx + 1}\n(Red: Parsed | Green: Solved)")
                axes[1].axis("off")

                plt.tight_layout()
                print(f"\n[INFO] Displaying complete results for Grid #{idx + 1}.")
                print("[INFO] CLOSE this plot window to view the next grid in sequence...")
                plt.show(block=True)
        except Exception as e:
            print(f"Error rendering terminal dashboards: {e}")


def main():
    """Main function used to load the images source directory"""
    folder_path = r"Project Test Cases\*.jpg"
    image_paths = sorted(glob.glob(folder_path))
    
    if not image_paths:
        print("No images found! Please check your directory path.")
        return
    else:
        print(f"Found {len(image_paths)} images. Loading Sequential Pipeline...")
        Project_MS_1(image_paths)
        print("All images processed successfully!")

if __name__ == "__main__":
    main()