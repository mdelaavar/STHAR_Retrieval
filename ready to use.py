import os
import shutil

def move_avi_files(folder_path):
    avi_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".avi"):
                source_path = os.path.join(root, file)
                destination_path = os.path.join(folder_path, file)
                shutil.move(source_path, destination_path)
                avi_files.append(destination_path)
                print(f"Moved: {file} to {folder_path}")

    # Remove all directories in the specified folder
    for dir_path in os.listdir(folder_path):
        full_path = os.path.join(folder_path, dir_path)
        if os.path.isdir(full_path):
            shutil.rmtree(full_path)
            print(f"Removed directory: {full_path}")

    return avi_files

# Replace 'your_folder_path' with the path to the folder you want to process
folder_path = 'qery/'
moved_files = move_avi_files(folder_path)

print("\nList of moved .avi files:")
for file in moved_files:
    print(file)
