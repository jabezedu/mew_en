import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from typing import Tuple, Optional

def preprocess_image(image_path: str) -> Optional[np.ndarray]:
    """
    Loads an image, converts it to grayscale, and applies an inverted
    binary threshold to isolate bright lines on a dark background.

    Args:
        image_path (str): Path to the input image file.

    Returns:
        Optional[np.ndarray]: The thresholded image as a NumPy array, or None if the image can't be loaded.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Use adaptive thresholding for better results on varying lighting, or simple binary
    # _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    # Using a slightly lower threshold to be more robust
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    # --- Noise Reduction ---
    # Use morphological opening to remove small noise artifacts
    kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return opened

def extract_line_data(thresh_img: np.ndarray, x_range: Tuple[float, float], y_range: Tuple[float, float]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Extracts the (x, y) coordinates of a line from a thresholded image.
    It calculates the center of the line for each column to get a clean curve.

    Args:
        thresh_img (np.ndarray): The preprocessed (thresholded) image.
        x_range (Tuple[float, float]): The minimum and maximum values of the x-axis (e.g., (0, 48) for hours).
        y_range (Tuple[float, float]): The minimum and maximum values of the y-axis (e.g., (0, 100) for SOC %).

    Returns:
        Optional[Tuple[np.ndarray, np.ndarray]]: A tuple containing the hours and SOC arrays, or None if no line is found.
    """
    h, w = thresh_img.shape
    hours, soc = [], []

    # --- Accurate Line Extraction ---
    # Find the mean y-coordinate for each x-coordinate (column)
    for x_pixel in range(w):
        y_pixels = np.where(thresh_img[:, x_pixel] > 0)[0]
        if y_pixels.any():
            # Average the y-pixel coordinates to find the center of the line
            avg_y_pixel = np.mean(y_pixels)
            
            # Convert pixel coordinates to data coordinates
            current_hour = (x_pixel / w) * (x_range[1] - x_range[0]) + x_range[0]
            # Flip y-axis (higher pixel value is lower SOC) and scale
            current_soc = y_range[1] - ((avg_y_pixel / h) * (y_range[1] - y_range[0]))
            
            hours.append(current_hour)
            soc.append(current_soc)

    if not hours:
        print("Error: No data points were extracted. Check image thresholding.")
        return None
        # This means no line was detected

    return np.array(hours), np.array(soc)

def calculate_kpis(hours: np.ndarray, soc: np.ndarray) -> dict:
    """
    Computes Key Performance Indicators (KPIs) from the SOC data.

    Args:
        hours (np.ndarray): The time data array.
        soc (np.ndarray): The State of Charge data array.

    Returns:
        dict: A dictionary containing the calculated KPIs.
    """
    # --- Accurate Time-Weighted Average ---
    # Use trapezoidal rule for a more accurate average than np.mean()
    time_weighted_avg = np.trapezoid(soc, x=hours) / (hours[-1] - hours[0])

    kpis = {
        "Max SOC": np.max(soc),
        "Min SOC": np.min(soc),
        "Time-Weighted Average SOC": time_weighted_avg,
        "Final SOC": soc[-1]
    }
    return kpis

def plot_results(image_path: str, hours: np.ndarray, soc: np.ndarray, x_range: Tuple[float, float], y_range: Tuple[float, float]):
    """
    Generates and displays a plot validating the extracted data against the original image.

    Args:
        image_path (str): Path to the original image for background.
        hours (np.ndarray): The extracted time data.
        soc (np.ndarray): The extracted SOC data.
        x_range (Tuple[float, float]): The x-axis range for plotting.
        y_range (Tuple[float, float]): The y-axis range for plotting.
    """
    original_img = plt.imread(image_path)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # --- Enhanced Validation Plot ---
    # Display the original chart as the background
    ax.imshow(original_img, aspect='auto', extent=(x_range[0], x_range[1], y_range[0], y_range[1]))
    
    # Overlay the extracted data points
    ax.plot(hours, soc, color='red', linewidth=2, label='Extracted Data')
    
    ax.set_xlabel("Hours", fontsize=12)
    ax.set_ylabel("SOC (%)", fontsize=12)
    ax.set_title("Validation: Extracted SOC Curve Overlaid on Original Chart", fontsize=14)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.show()

def main():
    """Main function to run the chart digitizer."""
    parser = argparse.ArgumentParser(description="Extracts SOC data from a chart image and computes KPIs.")
    parser.add_argument("image_path", type=str, help="Path to the SOC chart image file.")
    parser.add_argument("--x_min", type=float, default=0.0, help="Minimum value of the x-axis (e.g., hours).")
    parser.add_argument("--x_max", type=float, default=48.0, help="Maximum value of the x-axis (e.g., hours).")
    parser.add_argument("--y_min", type=float, default=0.0, help="Minimum value of the y-axis (e.g., SOC %%).")
    parser.add_argument("--y_max", type=float, default=100.0, help="Maximum value of the y-axis (e.g., SOC %%).")
    
    args = parser.parse_args()

    # 1. Preprocess the image
    processed_img = preprocess_image(args.image_path)
    if processed_img is None:
        return

    # 2. Extract the data
    data = extract_line_data(processed_img, (args.x_min, args.x_max), (args.y_min, args.y_max))
    if data is None:
        return
    hours, soc = data

    # 3. Compute KPIs
    kpis = calculate_kpis(hours, soc)
    
    # 4. Print the report
    print("\n--- KPI Report ---")
    for key, value in kpis.items():
        print(f"{key}: {value:.2f}%")
    print("------------------\n")

    # 5. Plot for visual verification
    plot_results(args.image_path, hours, soc, (args.x_min, args.x_max), (args.y_min, args.y_max))

if __name__ == "__main__":
    main()