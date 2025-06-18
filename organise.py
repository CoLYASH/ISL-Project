import os
import shutil

# Define paths
source_folder = r"C:\Users\Yash Dhoney\PycharmProjects\pythonProject\ISL_CSLRT_Corpus new\ISL_CSLRT_Corpus\INDIAN SIGN LANGUAGE ANIMATED VIDEOS"
destination_folder = r"C:\Users\Yash Dhoney\PycharmProjects\pythonProject\ISL_CSLRT_Corpus new\ISL_CSLRT_Corpus\new_ones"

# Create categorized folders
categories = ["Letters", "Words", "Numbers"]
for category in categories:
    os.makedirs(os.path.join(destination_folder, category), exist_ok=True)

# Organize files
for filename in os.listdir(source_folder):
    file_path = os.path.join(source_folder, filename)

    if not os.path.isfile(file_path):
        continue

    # Determine category
    if filename[0].isdigit():  # Check if the filename starts with a number
        target_folder = "Numbers"
    elif len(filename) == 1 and filename.isalpha():  # Single letters
        target_folder = "Letters"
    else:  # Default to words
        target_folder = "Words"

    # Move file
    shutil.move(file_path, os.path.join(destination_folder, target_folder, filename))

print("Files organized successfully!")
