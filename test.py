import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
img = cv2.imread("image.png")

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Threshold to isolate the line
_, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

# Find coordinates of the line
coords = np.column_stack(np.where(thresh > 0))

# Flip y-axis so higher values = higher SOC
h = img.shape[0]
coords[:,0] = h - coords[:,0]

# Normalize SOC (%) from y-coordinates
soc = (coords[:,0] / h) * 100

# Hours from x-coordinates
w = img.shape[1]
hours = (coords[:,1] / w) * 48  # since x-axis = 0 → 48 hours

# Compute KPIs
max_soc = np.max(soc)
min_soc = np.min(soc)
avg_soc = np.mean(soc)
final_soc = soc[-1]

print("KPI Report")
print("----------")
print(f"Max SOC: {max_soc:.2f}%")
print(f"Min SOC: {min_soc:.2f}%")
print(f"Average SOC: {avg_soc:.2f}%")
print(f"Final SOC: {final_soc:.2f}%")

# Plot extracted curve to verify
plt.plot(hours, soc, ".", markersize=1)
plt.xlabel("Hours")
plt.ylabel("SOC (%)")
plt.title("Extracted SOC Curve")
plt.show()
