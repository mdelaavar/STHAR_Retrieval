# Copy the dataset folder in the same directory as the Python code and modify line 18 of the code.
# Copy a folder of oquery videos in the same directory.  contains query videos with unknown names.
# When the code asks you for its address, enter its name with a slash like this "Query-folder/"

import os
import cv2
import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

# Define the root folder path where videos are stored
dataset_folder_path = 'UCF069/'

# Step 1: Split videos into training and testing sets
video_files = [file for file in os.listdir(dataset_folder_path) if file.endswith(".avi")]

activity_labels = [file.split("_")[0] for file in video_files]

train_videos, test_videos, train_labels, test_labels = train_test_split(
        video_files, activity_labels, test_size=0.2, random_state=42)  # , stratify=activity_labels)

label_mapping = {label: idx for idx, label in enumerate(set(activity_labels))}
print(label_mapping)

# Define paths for training and testing folders
train_folder_path = os.path.join(dataset_folder_path, 'train')
test_folder_path = os.path.join(dataset_folder_path, 'test')
train_process_path = os.path.join(dataset_folder_path, 'train-doc')
test_process_path = os.path.join(dataset_folder_path, 'test-doc')

# Create training and testing folders if they do not exist
os.makedirs(train_folder_path, exist_ok=True)
os.makedirs(test_folder_path, exist_ok=True)
os.makedirs(train_process_path, exist_ok=True)
os.makedirs(test_process_path, exist_ok=True)


# Move videos to training and testing folders
def move_videos(destination_folder, video_files):
        for video_file in video_files:
                source_path = os.path.join(dataset_folder_path, video_file)
                destination_path = os.path.join(destination_folder, video_file)
                os.rename(source_path, destination_path)


move_videos(train_folder_path, train_videos)
move_videos(test_folder_path, test_videos)

# Step 2: Process videos and extract frames
frames_per_gop = 30


def process_videos(dataset_path, process_path):
        video_files = [file for file in os.listdir(dataset_path) if file.endswith(".avi")]
        for video_file in video_files:
                print(video_file)
                video_path = os.path.join(dataset_path, video_file)
                activity_name = video_file.split("_")[0]

                cap = cv2.VideoCapture(video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                print(total_frames)
                gops_count = math.ceil(total_frames / frames_per_gop)
                print(gops_count)

                activity_folder_path = os.path.join(process_path, activity_name)
                os.makedirs(activity_folder_path, exist_ok=True)

                video_folder_path = os.path.join(activity_folder_path, video_file.split(".")[0])
                os.makedirs(video_folder_path, exist_ok=True)

                for i in range(gops_count):
                        gop_folder_path = os.path.join(video_folder_path, f"GOP_{i}")
                        os.makedirs(gop_folder_path, exist_ok=True)

                        for frame_num in range(frames_per_gop):
                                ret, frame = cap.read()
                                if not ret:
                                        break
                                frame_path = os.path.join(gop_folder_path, f"{activity_name}{i}{frame_num}.jpg")
                                cv2.imwrite(frame_path, frame)

                cap.release()


# Process training and testing videos
process_videos(train_folder_path, train_process_path)
process_videos(test_folder_path, test_process_path)


# Step 3: Save keyframes
def save_keyframes(dataset_process_path):
        activity_folders = [file for file in os.listdir(dataset_process_path) if
                            os.path.isdir(os.path.join(dataset_process_path, file))]
        for activity_folder in activity_folders:
                activity_folder_path = os.path.join(dataset_process_path, activity_folder)
                video_folders = [file for file in os.listdir(activity_folder_path) if
                                 os.path.isdir(os.path.join(activity_folder_path, file))]
                for video_folder in video_folders:
                        gop_folders = [file for file in
                                       os.listdir(os.path.join(dataset_process_path, activity_folder, video_folder)) if
                                       os.path.isdir(
                                               os.path.join(dataset_process_path, activity_folder, video_folder, file))]
                        for gop_folder in gop_folders:
                                frames_folder_path = os.path.join(dataset_process_path, activity_folder, video_folder,
                                                                  gop_folder)
                                frame_files = [file for file in os.listdir(frames_folder_path) if file.endswith(".jpg")]
                                if frame_files:
                                        keyframe = frame_files[0]
                                        keyframe_path = os.path.join(frames_folder_path, f"{gop_folder}_keyframe.jpg")
                                        os.rename(os.path.join(frames_folder_path, keyframe), keyframe_path)


# Save keyframes for training and testing videos
save_keyframes(train_process_path)
save_keyframes(test_process_path)


# Step 4: Extract spatial features
def extract_spatial_features(frame_path):
        frame = cv2.imread(frame_path)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (128, 128))
        features = np.reshape(resized, (128, 128))  # Reshape to a 4D tensor
        return features


def extract_temporal_features(frame_path1, frame_path2):
        frame1 = cv2.imread(frame_path1, cv2.IMREAD_GRAYSCALE)
        frame2 = cv2.imread(frame_path2, cv2.IMREAD_GRAYSCALE)
        frame_diff = cv2.absdiff(frame1, frame2)
        return frame_diff


def extract_temporal_features_from_meis(gop_folder_path):
        frame_files = [file for file in os.listdir(gop_folder_path) if file.endswith(".jpg")]
        frame_files.sort()  # Ensure frames are in the correct order
        frame_differences = []
        # Check if the number of frames is less than the minimum required (2 or 3)
        min_required_frames = 2  # You can adjust this to 3 if you find it more appropriate
        if len(frame_files) < min_required_frames:
                for i in range(len(frame_files)):
                        frame_path1 = os.path.join(gop_folder_path, frame_files[i])
                        frame_path2 = os.path.join(gop_folder_path, frame_files[i])
                        frame_diff = extract_temporal_features(frame_path1, frame_path2)
                        frame_differences.append(frame_diff)
        else:
                for i in range(len(frame_files) - 1):
                        frame_path1 = os.path.join(gop_folder_path, frame_files[i])
                        frame_path2 = os.path.join(gop_folder_path, frame_files[i + 1])
                        frame_diff = extract_temporal_features(frame_path1, frame_path2)
                        frame_differences.append(frame_diff)

        # Convert to a 3D numpy array
        g_temporal_features = np.array(frame_differences)
        return g_temporal_features


def aggregate_features_std(features_list):
        return np.std(features_list, axis=0)


def extract_spatial_temporal_features_from_keyframes(process_folder_path):
        global gop_temporal_features, video_temporal_features, video_spatial_features, video_folder_path, temporal_folder_path, spatial_folder_path
        activity_folders = [file for file in os.listdir(process_folder_path) if
                            os.path.isdir(os.path.join(process_folder_path, file))]

        for activity_folder in activity_folders:
                activity_folder_path = os.path.join(process_folder_path, activity_folder)
                video_folders = [file for file in os.listdir(activity_folder_path) if
                                 os.path.isdir(os.path.join(activity_folder_path, file))]

                spatial_folder_path = os.path.join(process_folder_path, "spatial_features")
                os.makedirs(spatial_folder_path, exist_ok=True)

                temporal_folder_path = os.path.join(process_folder_path, "temporal_features")
                os.makedirs(temporal_folder_path, exist_ok=True)

                for video_folder in video_folders:
                    print(video_folder)
                    gop_spatial_features = []  # List to store all spatial feature vectors of each video file
                    video_folder_path = os.path.join(activity_folder_path, video_folder)
                    gop_folders = [file for file in os.listdir(video_folder_path)]

                    i = 0
                    for gop_folder in gop_folders:
                            gop_folder_path = os.path.join(video_folder_path, gop_folder)
                            spatial_path = os.path.join(video_folder_path, f'spatial_{i}')
                            temporal_path = os.path.join(video_folder_path, f'temporal_{i}')

                            keyframe_file = [file for file in os.listdir(gop_folder_path) if
                                                file.endswith("_keyframe.jpg")]
                            keyframe_path = os.path.join(gop_folder_path, keyframe_file[0])

                            gop_spatial_feature = extract_spatial_features(keyframe_path)
                            #                np.save(spatial_path, gop_spatial_feature)
                            #
                            gop_spatial_features.append(gop_spatial_feature)

                            gop_temporal_features = extract_temporal_features_from_meis(gop_folder_path)
                            #                print(f"gop_temporal size: {gop_temporal_features.shape}")
                            #                np.save(temporal_path, gop_temporal_features)
                            if i == 1:
                                    print(f'gop_spatial_feature: {gop_spatial_feature}')
                                    # print(f'gop_temporal_features: {gop_temporal_features}')
                                    print(f'gop_spatial_size: {gop_spatial_feature.shape}')
                                    # print(f'gop_temporal_size: {gop_temporal_features.shape}')
                            i += 1

                    # print(f'list of gop spatial featurs {video_folder} is {gop_spatial_features}')
                    video_spatial_features = aggregate_features_std(gop_spatial_features)
                    np.save(os.path.join(spatial_folder_path, f"spatial_features_{video_folder}"),video_spatial_features)
                    # print(f'video_spatial_features: {video_spatial_features}')

                    # print(f'list of gop temporal featurs {video_folder} is {gop_temporal_features}')
                    video_temporal_features = aggregate_features_std(gop_temporal_features)
                    np.save(os.path.join(temporal_folder_path, f"temporal_features_{video_folder}"),video_temporal_features)
                    # print(f'video_temporal_features: {video_temporal_features}')

        return video_spatial_features, video_temporal_features,


# Extract and save  spatial and temporal features for training and testing videos
extract_spatial_temporal_features_from_keyframes(train_process_path)
extract_spatial_temporal_features_from_keyframes(test_process_path)

print(f"spatial size: {video_spatial_features.shape}")
print(f"temporal size: {video_temporal_features.shape}")


# Step 5: Training model

# ConvNet for Spatial Features
class SpatialConvNet(nn.Module):
        def __init__(self, num_classes):
                super(SpatialConvNet, self).__init__()
                self.layer1 = self._make_layer(1, 64)  # 1 input channel for grayscale
                self.layer2 = self._make_layer(64, 128)
                self.layer3 = self._make_layer(128, 256)
                self.fc = nn.Linear(256, num_classes)  # Output features based on num_classes

        def _make_layer(self, in_channels, out_channels):
                layer = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True)
                )
                return layer

        def forward(self, x):
                out = self.layer1(x)
                out = self.layer2(out)
                out = self.layer3(out)
                out = F.adaptive_avg_pool2d(out, (1, 1))
                out = out.view(out.size(0), -1)
                out = self.fc(out)
                return out


# ConvNet for Temporal Features
class TemporalConvNet(nn.Module):
        def __init__(self, num_classes):
                super(TemporalConvNet, self).__init__()
                self.layer1 = self._make_layer(1, 64)  # 1 input channel for grayscale
                self.layer2 = self._make_layer(64, 128)
                self.layer3 = self._make_layer(128, 256)
                self.fc = nn.Linear(256, num_classes)  # Output features based on num_classes

        #                self.fc = nn.Linear(256, 5)  # Adjust the output features to 5

        def _make_layer(self, in_channels, out_channels):
                layer = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True)
                )
                return layer

        def forward(self, x):
                out = self.layer1(x)
                out = self.layer2(out)
                out = self.layer3(out)
                out = F.adaptive_avg_pool2d(out, (1, 1))
                out = out.view(out.size(0), -1)
                out = self.fc(out)

                return out


# (new) Attention Mechanism for Temporal Features
class Attention(nn.Module):
        def __init__(self, input_size, hidden_size):
                super(Attention, self).__init__()
                self.W_query = nn.Linear(input_size, hidden_size)
                self.W_key = nn.Linear(input_size, hidden_size)
                self.W_value = nn.Linear(input_size, hidden_size)
                self.softmax = nn.Softmax(dim=-1)

        def forward(self, temporal_outputs, spatial_outputs):
                query = self.W_query(temporal_outputs)
                key = self.W_key(spatial_outputs)
                value = self.W_value(spatial_outputs)

                # Calculate attention scores
                attention_scores = torch.matmul(query, key.transpose(-2, -1))
                attention_scores = F.softmax(attention_scores, dim=-1)

                # Apply attention to values
                attended_values = torch.matmul(attention_scores, value)

                return attended_values


# Dataset Class for Loading Features and Labels
class VideoDataset(Dataset):
        def __init__(self, spatial_features_folder, temporal_features_folder, label_mapping):
                self.spatial_features = []
                self.temporal_features = []
                self.labels = []

                # Ensure the same ordering of files in both folders
                spatial_files = sorted(os.listdir(spatial_features_folder))
                temporal_files = sorted(os.listdir(temporal_features_folder))

                for spatial_file, temporal_file in zip(spatial_files, temporal_files):
                        if spatial_file.endswith('.npy') and temporal_file.endswith('.npy'):
                                # Load features
                                spatial_feature = np.load(os.path.join(spatial_features_folder, spatial_file))
                                temporal_feature = np.load(os.path.join(temporal_features_folder, temporal_file))

                                # Normalize features
                                spatial_feature = spatial_feature.astype(np.float32) / 255.0
                                temporal_feature = temporal_feature.astype(np.float32) / 255.0

                                # Extract label from file name and map to integer
                                label_name = spatial_file.split('_')[
                                        2]  # Adjust based on how your file names are structured
                                label_index = label_mapping.get(label_name, -1)  # Default to -1 for unknown labels

                                # Append to lists if label found
                                if label_index != -1:
                                        self.spatial_features.append(spatial_feature)
                                        self.temporal_features.append(temporal_feature)
                                        self.labels.append(label_index)

                # Convert lists to tensors
                self.spatial_features = torch.tensor(self.spatial_features).unsqueeze(1)
                self.temporal_features = torch.tensor(self.temporal_features).unsqueeze(1)
                self.labels = torch.tensor(self.labels, dtype=torch.long)

        def __len__(self):
                return len(self.spatial_features)

        def __getitem__(self, idx):
                return self.spatial_features[idx], self.temporal_features[idx], self.labels[idx]


# Assuming hidden_size is the same as the size of spatial_outputs and temporal_outputs
hidden_size = 5

# Initialize Attention mechanism
attention = Attention(input_size=hidden_size, hidden_size=hidden_size)

# Instantiate the Networks
num_classes = len(label_mapping)
spatial_conv_net = SpatialConvNet(num_classes=num_classes)

temporal_conv_net = TemporalConvNet(num_classes=num_classes)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
        list(spatial_conv_net.parameters()) +
        list(temporal_conv_net.parameters()) +
        list(attention.parameters()),
        lr=0.001
)
print(f'the spatial path befor train: {spatial_folder_path}')
print(f'the temporal path befor train: {temporal_folder_path}')
train_dataset = VideoDataset(spatial_folder_path, temporal_folder_path, label_mapping)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Training loop
num_epochs = 10  # Adjust as needed
for epoch in range(num_epochs):
        print(f"step of train loop: {epoch}")
        for spatial_features, temporal_features, labels in train_loader:
                # Forward pass
                spatial_outputs = spatial_conv_net(spatial_features)
                temporal_outputs = temporal_conv_net(temporal_features)
                temporal_attention_outputs = attention(temporal_outputs, spatial_outputs)

                combined_output = spatial_outputs + temporal_outputs
                # combined_output = spatial_outputs + temporal_attention_outputs
                # Compute loss
                loss = criterion(combined_output, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

"""
       **********************************************************************
                     PART 2: Retrieval Query Video File
       **********************************************************************
"""

# Step 6: Retrieval

video_query_path = input("ENTER THE DIRECTORY OF THE QUERY VIDEO FILE: ")
video_query_files = [file for file in os.listdir(video_query_path) if file.endswith(".avi")]

# making GOP folders and save frames in GOP folders of query videos
frames_per_gop = 30
for Qvideo_file in video_query_files:
        Qvideo_path = os.path.join(video_query_path, Qvideo_file)
        Qvideo_folder_path = os.path.join(video_query_path, Qvideo_file.split(".")[0])
        os.makedirs(Qvideo_folder_path, exist_ok=True)
        cap = cv2.VideoCapture(Qvideo_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        gops_count = math.ceil(total_frames / frames_per_gop)

        for i in range(gops_count):
                gop_folder_path = os.path.join(Qvideo_folder_path, f"GOP_{i}")
                os.makedirs(gop_folder_path, exist_ok=True)

                for frame_num in range(frames_per_gop):
                        ret, frame = cap.read()
                        if not ret:
                                break
                        frame_path = os.path.join(gop_folder_path, f"{frame_num}.jpg")
                        cv2.imwrite(frame_path, frame)

        cap.release()

# Making keyframes
video_query_folders = [file for file in os.listdir(video_query_path) if
                       os.path.isdir(os.path.join(video_query_path, file))]
for Qvideo_folder in video_query_folders:
        gop_folders = [file for file in os.listdir(os.path.join(video_query_path, Qvideo_folder)) if
                       os.path.isdir(os.path.join(video_query_path, Qvideo_folder, file))]
        for gop_folder in gop_folders:
                frames_folder_path = os.path.join(video_query_path, Qvideo_folder, gop_folder)
                frame_files = [file for file in os.listdir(frames_folder_path) if file.endswith(".jpg")]
                if frame_files:
                        keyframe = frame_files[0]
                        keyframe_path = os.path.join(frames_folder_path, f"{gop_folder}_keyframe.jpg")
                        os.rename(os.path.join(frames_folder_path, keyframe), keyframe_path)


# Extract spatial/temporal features of query files

def extract_spatial_features(frame_path):
        frame = cv2.imread(frame_path)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (128, 128))
        features = np.reshape(resized, (128, 128))  # Reshape to a 4D tensor
        return features


def extract_temporal_features(frame_path1, frame_path2):
        frame1 = cv2.imread(frame_path1, cv2.IMREAD_GRAYSCALE)
        frame2 = cv2.imread(frame_path2, cv2.IMREAD_GRAYSCALE)
        frame_diff = cv2.absdiff(frame1, frame2)
        return frame_diff


def extract_temporal_features_from_meis(gop_folder_path):
        frame_files = [file for file in os.listdir(gop_folder_path) if file.endswith(".jpg")]
        frame_files.sort()  # Ensure frames are in the correct order
        frame_differences = []
        # Check if the number of frames is less than the minimum required (2 or 3)
        min_required_frames = 2  # You can adjust this to 3 if you find it more appropriate
        if len(frame_files) < min_required_frames:
                for i in range(len(frame_files)):
                        frame_path1 = os.path.join(gop_folder_path, frame_files[i])
                        frame_path2 = os.path.join(gop_folder_path, frame_files[i])
                        frame_diff = extract_temporal_features(frame_path1, frame_path2)
                        frame_differences.append(frame_diff)
        else:
                for i in range(len(frame_files) - 1):
                        frame_path1 = os.path.join(gop_folder_path, frame_files[i])
                        frame_path2 = os.path.join(gop_folder_path, frame_files[i + 1])
                        frame_diff = extract_temporal_features(frame_path1, frame_path2)
                        frame_differences.append(frame_diff)

        # Convert to a 3D numpy array
        g_temporal_features = np.array(frame_differences)
        return g_temporal_features


def aggregate_features_std(features_list):
        return np.std(features_list, axis=0)


for Qvideo_folder in video_query_folders:
        Qvideo_folder_path = os.path.join(video_query_path, Qvideo_folder)
        gop_spatial_features = []

        spatial_folder_path = os.path.join(Qvideo_folder_path, "spatial_features")
        os.makedirs(spatial_folder_path, exist_ok=True)

        temporal_folder_path = os.path.join(Qvideo_folder_path, "temporal_features")
        os.makedirs(temporal_folder_path, exist_ok=True)

        gop_folders = [file for file in os.listdir(Qvideo_folder_path) if file.startswith("GOP")]

        i = 0
        for gop_folder in gop_folders:
                gop_folder_path = os.path.join(Qvideo_folder_path, gop_folder)
                spatial_path = os.path.join(Qvideo_folder_path, f'spatial_{i}')
                temporal_path = os.path.join(Qvideo_folder_path, f'temporal_{i}')

                keyframe_file = [file for file in os.listdir(gop_folder_path) if file.startswith("GOP")]
                keyframe_path = os.path.join(gop_folder_path, keyframe_file[0])

                gop_spatial_feature = extract_spatial_features(keyframe_path)
                np.save(spatial_path, gop_spatial_feature)
                gop_spatial_features.append(gop_spatial_feature)

                gop_temporal_features = extract_temporal_features_from_meis(gop_folder_path)
                #                print(f"gop_temporal size: {gop_temporal_features.shape}")
                np.save(temporal_path, gop_temporal_features)
                i += 1

        Qvideo_spatial_features = aggregate_features_std(gop_spatial_features)
        np.save(os.path.join(spatial_folder_path, f"spatial_features_{Qvideo_folder}"), Qvideo_spatial_features)

        Qvideo_temporal_features = aggregate_features_std(gop_temporal_features)
        np.save(os.path.join(temporal_folder_path, f"temporal_features_{Qvideo_folder}"), Qvideo_temporal_features)

# return Qvideo_spatial_features, Qvideo_temporal_features,


# feature loading

def classify_query_video(spatial_features_path, temporal_features_path, spatial_conv_net, temporal_conv_net): # , attention):
        # Load and preprocess spatial features
        spatial_features = np.load(spatial_features_path)
        spatial_features = torch.tensor(spatial_features.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)

        # Load and preprocess temporal features
        temporal_features = np.load(temporal_features_path)
        temporal_features = torch.tensor(temporal_features.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)

        # Forward pass through the models
        with torch.no_grad():  # No gradient needed
                spatial_output = spatial_conv_net(spatial_features)
                temporal_output = temporal_conv_net(temporal_features)
                combined_output = spatial_output + temporal_output
                # combined_output = attention(spatial_output + temporal_output) 
                
                #combined_output = attention(temporal_output,
                #                            spatial_output) if attention else spatial_output + temporal_output

        return combined_output


# Example usage

for Qvideo_folder in video_query_folders:
        query_spatial_features_path = os.path.join(video_query_path, Qvideo_folder, 'spatial_features', f"spatial_features_{Qvideo_folder}.npy")
        query_temporal_features_path = os.path.join(video_query_path, Qvideo_folder, 'temporal_features', f"temporal_features_{Qvideo_folder}.npy")
        combined_output = classify_query_video(query_spatial_features_path, query_temporal_features_path,
                                               spatial_conv_net, temporal_conv_net) # , attention)

        # Apply softmax if the final layer of your model is not softmax
        probabilities = F.softmax(combined_output, dim=1)
        predicted_label_idx = torch.argmax(probabilities, dim=1).item()

        # Map the predicted index to the actual label
        predicted_label = [label for label, idx in label_mapping.items() if idx == predicted_label_idx][0]
        print(f"Predicted label for '{Qvideo_folder}': {predicted_label}")

        # Optional: Code to find videos from the original dataset with the same label
