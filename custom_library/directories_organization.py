import os

def replicate_directory_structure(source_directory, target_directory):
    '''Walk through the source directory and create the same structure in the target directory. It returns the list of subdirectories paths.'''

    subdirectories_paths_list = list()
    for root, dirs, files in os.walk(source_directory):
        relative_path = os.path.relpath(root, source_directory)
        target_path = os.path.join(target_directory, relative_path)

        if not os.path.exists(target_path):
            os.makedirs(target_path)

        # Append to the list only subdirectories without any further subdirectories or files
        if not dirs and files:
            subdirectories_paths_list.append(target_path)
        
    return subdirectories_paths_list


def list_leaf_subdirectories(directory):
    '''Return a list of all the subdirectories excluding those with other further subdirectories and files.'''

    leaf_subdirectories = []
    for root, dirs, files in os.walk(directory):
        if not dirs and files:
            leaf_subdirectories.append(root)

    return leaf_subdirectories

    
def change_root(old_path, old_root, new_root):
    # Ensure old_path starts with old_root
    if not old_path.startswith(old_root):
        raise ValueError("old_path does not start with old_root")

    # Remove the old_root from the beginning of old_path
    relative_path = old_path[len(old_root):]

    # Create the new path by joining new_root and the relative path
    new_path = os.path.join(new_root, relative_path)

    return new_path
