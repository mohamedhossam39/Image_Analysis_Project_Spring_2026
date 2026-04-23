import numpy as np
import cv2
import glob
from matplotlib import pyplot as plt
import logging

class Project_MS_1:
    def __init__(self, image_paths):
        """
        Initializes the class by loading a list of image paths into a list of image arrays.
        """
        self.images = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]

        for idx, img in enumerate(self.images):
            if img is None:
                print(f"Warning: Failed to load image at index {idx}. Check the file path.")
        self.show_images_grid()

    def show_images_grid(self):
        """Display all 16 images in a 4x4 grid using Matplotlib."""
        try:
            plt.figure(figsize=(12, 12))
            for idx, img in enumerate(self.images):
                plt.subplot(4, 4, idx + 1)
                plt.imshow(img, cmap='gray')
                plt.title(f"Image {idx + 1}")
                plt.axis('off')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"error showing image grid {e}")
        finally:
            self.pre_process_images()

    def pre_process_images(self):
        try:
            self.processed_images = [] 
            plt.figure(figsize=(12, 12))

            for idx, img in enumerate(self.images):
                img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
                blurred = cv2.bilateralFilter(img_norm, 9, 75, 75)
                blurred = cv2.GaussianBlur(blurred, (5, 5), 0)
                thresh = cv2.adaptiveThreshold(
                    blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv2.THRESH_BINARY_INV, 11, 2
                )
                kernel = np.ones((3, 3), np.uint8)
                dilated = cv2.dilate(thresh, kernel, iterations=2)
                processed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
                self.processed_images.append(processed)
                plt.subplot(4, 4, idx + 1)
                plt.imshow(processed, cmap='gray')
                plt.title(f"Image {idx + 1}")
                plt.axis('off')
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Error in preprocessing: {e}")
        finally:
            self.isolate_outer_frame()

    def isolate_outer_frame(self):
            try:
                self.framed_images = []
                self.corners_list = [] 

                plt.figure(figsize=(12, 12))

                for idx, img in enumerate(self.processed_images):
                    if img[0, 0] > 127:
                        contour_img = cv2.bitwise_not(img)
                    else:
                        contour_img = img.copy()

                    contours, _ = cv2.findContours(contour_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    display_img = cv2.cvtColor(self.images[idx], cv2.COLOR_GRAY2BGR)

                    grid_corners = None

                    if contours:

                        contours = sorted(contours, key=cv2.contourArea, reverse=True)

                        for cnt in contours:
                            perimeter = cv2.arcLength(cnt, True)

                            approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)

                            if len(approx) == 4:
                                grid_corners = approx
                                cv2.drawContours(display_img, [grid_corners], -1, (0, 255, 0), 5)

                                for point in grid_corners:
                                    x, y = point.ravel()
                                    cv2.circle(display_img, (x, y), 8, (0, 0, 255), -1)
                                break

                    self.corners_list.append(grid_corners)
                    self.framed_images.append(display_img)

                    plt.subplot(4, 4, idx + 1)
                    plt.imshow(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB))
                    plt.title(f"Image {idx + 1}")
                    plt.axis('off')

                plt.tight_layout()
                plt.show()

            except Exception as e:
                print(f"error isolating frame {e}")
            finally:

                self.straighten_grids()

    def straighten_grids(self):
        try:
            self.straightened_images = []
            plt.figure(figsize=(12, 12))

            for idx, corners in enumerate(self.corners_list):
                original_img = self.images[idx]

                if corners is not None:

                    pts = corners.reshape(4, 2)
                    rect = np.zeros((4, 2), dtype="float32")

                    s = pts.sum(axis=1)
                    rect[0] = pts[np.argmin(s)] 

                    rect[2] = pts[np.argmax(s)] 

                    diff = np.diff(pts, axis=1)
                    rect[1] = pts[np.argmin(diff)] 

                    rect[3] = pts[np.argmax(diff)] 

                    side = 450
                    dst = np.array([
                        [0, 0],
                        [side - 1, 0],
                        [side - 1, side - 1],
                        [0, side - 1]], dtype="float32")

                    M = cv2.getPerspectiveTransform(rect, dst)
                    straightened = cv2.warpPerspective(original_img, M, (side, side))
                else:

                    straightened = original_img

                self.straightened_images.append(straightened)

                plt.subplot(4, 4, idx + 1)
                plt.imshow(straightened, cmap='gray')
                plt.title(f"Straightened {idx + 1}")
                plt.axis('off')

            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"error straightening grids {e}")

def main():
    """Main function used to load the images source directory"""
    folder_path = r"Project Test Cases\*.jpg" 
    image_paths = sorted(glob.glob(folder_path))
    if not image_paths:
        print("No images found! Please check your directory path.")
        return
    print(f"Found {len(image_paths)} images. Loading...")
    project = Project_MS_1(image_paths)
    print("All images loaded successfully!")

if __name__ == '__main__':
    main()