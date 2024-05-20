import pandas as pd
import os
import re

# File path to the Vicon CSV
file_path = 'Trial115.csv'

# Manually determined sync points (example: RealSense image and Vicon start frame)
manual_sync_realsense = 1715977380.105901  # example timestamp from RealSense
manual_sync_vicon_frame = 1351  # example frame number in Vicon

output_csv_path = 'synchronized_frames_16.csv'

# Load and process the Vicon CSV
column_labels = pd.read_csv(file_path, skiprows=4, nrows=2, header=None)
marker_names = column_labels.iloc[0].fillna(method='ffill')
axis_labels = column_labels.iloc[1]
multi_index = pd.MultiIndex.from_arrays([marker_names, axis_labels], names=('Marker', 'Axis'))
data = pd.read_csv(file_path, skiprows=7, header=None)
if len(multi_index) > data.shape[1]:
    multi_index = multi_index[:data.shape[1]]
data.columns = multi_index

# Extract the frame data
vicon_frames = data.loc[:, (float('nan'), 'Frame')].astype(int)

# Debug: Print unique frame numbers to check for issues
print("Unique Vicon frame numbers:", vicon_frames.unique())

# Filter out frames labeled as 0 and repeated frames
valid_frames_mask = (vicon_frames != 0) & (vicon_frames.shift() != vicon_frames)
vicon_frames = vicon_frames[valid_frames_mask].reset_index(drop=True)
data = data[valid_frames_mask].reset_index(drop=True)

# Determine the actual starting frame number
vicon_start_frame = vicon_frames.iloc[0]

# Debug: Print the starting frame number
print("Actual starting frame number in Vicon data:", vicon_start_frame)

# Parse timestamps from RealSense filenames and sort
image_files = os.listdir('realsense_frames17_20240517232252_color')
image_data = []
for filename in image_files:
    match = re.search(r'_(\d+\.\d+)\.png$', filename)
    if match:
        timestamp = float(match.group(1))
        image_data.append((filename, timestamp))
image_data.sort(key=lambda x: x[1])  # Sort by timestamp

# Calculate the Vicon start time assuming 100 Hz
vicon_start_time = manual_sync_realsense - ((manual_sync_vicon_frame - vicon_start_frame) / 100.0)

# Generate timestamps for Vicon frames
vicon_timestamps = [vicon_start_time + ((frame - vicon_start_frame) / 100.0) for frame in vicon_frames]

# Map RealSense frames to Vicon frames
mapped_frames = []
for filename, timestamp in image_data:
    # Find the nearest Vicon timestamp
    closest_vicon_frame_index = min(range(len(vicon_timestamps)), key=lambda i: abs(vicon_timestamps[i] - timestamp))
    
    # Check if the timestamp difference is within ±5 ms range
    if abs(vicon_timestamps[closest_vicon_frame_index] - timestamp) <= 0.005:
        # Adjust for XYZ coordinates being one frame ahead
        coord_frame_index = closest_vicon_frame_index + 1  # offset by +1
        if coord_frame_index < len(vicon_frames):  # check bounds
            frame_data = {
                'RealSense_frame': filename,
                'Vicon_frame': vicon_frames.iloc[closest_vicon_frame_index],  # Use the actual frame number
                'Vicon_timestamp': vicon_timestamps[closest_vicon_frame_index]
            }
            for marker in marker_names.dropna().unique():
                for axis in ['X', 'Y', 'Z']:
                    try:
                        frame_data[f'{marker}_{axis}'] = data.loc[coord_frame_index, (marker, axis)]
                    except KeyError:
                        frame_data[f'{marker}_{axis}'] = None  # Handle missing data gracefully
            mapped_frames.append(frame_data)
        else:
            print(f"XYZ coordinates for frame {coord_frame_index} are out of bounds.")
    else:
        print(f"RealSense frame {filename} not matched due to timestamp difference exceeding ±5 ms.")

# Convert mapped frames to DataFrame
mapped_frames_df = pd.DataFrame(mapped_frames)

# Save the DataFrame to a CSV file
mapped_frames_df.to_csv(output_csv_path, index=False)

# Confirmation
print(f"Data has been saved to {output_csv_path}")
