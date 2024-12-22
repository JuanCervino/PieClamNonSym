import os

def rename_files_in_directory(directory, old_str, new_str):
    for root, dirs, files in os.walk(directory, topdown=False):
        # Rename files
        for file_name in files:
            if old_str in file_name:
                new_file_name = file_name.replace(old_str, new_str)
                old_file_path = os.path.join(root, file_name)
                new_file_path = os.path.join(root, new_file_name)
                os.rename(old_file_path, new_file_path)
                print(f'Renamed file: {old_file_path} -> {new_file_path}')
        
        # Rename directories
        for dir_name in dirs:
            if old_str in dir_name:
                new_dir_name = dir_name.replace(old_str, new_str)
                old_dir_path = os.path.join(root, dir_name)
                new_dir_path = os.path.join(root, new_dir_name)
                os.rename(old_dir_path, new_dir_path)
                print(f'Renamed directory: {old_dir_path} -> {new_dir_path}')

# Replace 'your_project_directory' with the path to your project directory
# Replace 'old_str' with the word you want to replace
# Replace 'new_str' with the new word
rename_files_in_directory('/home/user/Documents/danny/AAAI_pieclam', 'iegam', 'ieclam')