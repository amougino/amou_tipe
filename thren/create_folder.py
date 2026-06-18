import os


def create_folder(path):
    try:
        os.mkdir(path)
    except FileExistsError:
        raise Exception(f"Directory '{path}' already exists.")
    except PermissionError:
        raise Exception(f"Permission denied: Unable to create '{path}'.")
    except Exception as e:
        raise Exception(f"An error occurred: {e}")
