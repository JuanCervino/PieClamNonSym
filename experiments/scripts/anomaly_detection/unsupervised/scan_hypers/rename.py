import os

def rename_files(directory):
    for filename in os.listdir(directory):
        if "mighty" in filename:
            new_filename = filename.replace("mighty", "global")
            os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))
            print(f'Renamed: {filename} -> {new_filename}')

if __name__ == "__main__":
    directory = "."  # Use the current directory
    rename_files(directory)