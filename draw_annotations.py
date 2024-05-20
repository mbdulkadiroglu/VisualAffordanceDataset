import os
import re
import cv2
import numpy as np
import pandas as pd

# Function to convert 3D pose to 2D pixel coordinates using camera intrinsics
def project_3d_to_2d(pose_3d, K):
    # Convert pose to homogeneous coordinates
    pose_3d_h = np.append(pose_3d, 1.0)
    # Project pose onto image plane using camera intrinsic matrix
    pixel_coords = np.dot(K, pose_3d_h[:3])
    # Normalize coordinates
    pixel_coords /= pixel_coords[2]
    pixel_coords = (int(pixel_coords[0]), int(pixel_coords[1]))
    return pixel_coords

# Function to transform Vicon coordinates to camera coordinates
def transform_vicon_to_camera(pose_vicon, R, t):
    pose_vicon = np.array(pose_vicon).reshape(-1, 3).T  # Reshape and transpose to (3, N)
    pose_cam = np.dot(R, pose_vicon) + t[:, np.newaxis]  # Correct broadcasting of t
    return pose_cam.T.reshape(-1)  # Transpose back and flatten

# Function to draw object annotations on an image
def draw_annotations(image, annotations, K, R, t, color=(0, 255, 0)):
    # Transform and draw all points
    all_points_transformed = transform_vicon_to_camera(annotations, R, t)
    for i in range(0, len(all_points_transformed), 3):
        x, y, z = all_points_transformed[i:i+3]
        if np.isnan([x, y, z]).any():
            continue  # Skip points with NaN values
        pixel_coords = project_3d_to_2d((x, y, z), K)
        cv2.circle(image, pixel_coords, 5, color, -1)  # Draw points in specified color

# Correct intrinsic parameters for Intel RealSense D435i
fx, fy, cx, cy, width, height = 605.8645629882812, 604.7568359375, 317.89990234375, 246.007568359375, 640, 480
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1]])

# Define the rotation matrix R and translation vector t
# These values need to be obtained from the calibration between Vicon and camera
R = np.array([[-0.90669318,  0.41477425, -0.07661462],  
              [0.10461526,  0.39710848,  -0.91178972],
              [-0.40861121, -0.81869847,  -0.40344727]])
t = np.array([290.289703, 986.314087, 1498.572021])

# Specify the directory containing the frames and the CSV file
frames_directory = 'realsense_frames17_20240517232252_color'
csv_file = 'synchronized_frames_16.csv'

# Load the CSV file
data = pd.read_csv(csv_file)

# Process frames
for index, row in data.iterrows():
    frame_file = row['RealSense_frame']
    print(f"Processing frame: {frame_file}")  # Debug print
    frame = cv2.imread(os.path.join(frames_directory, frame_file), cv2.IMREAD_COLOR)

    # Extract annotations
    annotations = row[3:].values.astype(float)  # Skip the first three columns (frame names and timestamp)

    # Draw annotations on the image
    draw_annotations(frame, annotations, K, R, t, color=(0, 255, 0))  # Use green color for all points

    # Save the output image
    output_filename = f"drawn_frames10/{os.path.splitext(frame_file)[0]}_output_obj_image.jpg"
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    cv2.imwrite(output_filename, frame)
    print("Output image saved as", output_filename)
