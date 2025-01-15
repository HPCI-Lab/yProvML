import gzip
import os
import shutil
import tarfile
from pathlib import Path

def compress_file(input_file, output_file):
    """Compress a file using gzip."""
    try:
        # File
        if Path(input_file).is_file():
            with open(input_file, 'rb') as f_in:
                with gzip.open(output_file + '.gz', 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            print(f"File '{input_file}' compressed to '{output_file}.gz'.")

        # Folder
        elif Path(input_file).is_dir():
            with tarfile.open(output_file + ".tar", "w") as tar:
                tar.add(output_file, arcname=os.path.basename(input_file))

            with open(output_file + ".tar", "rb") as f_in:
                with gzip.open(output_file + ".tar.gz", "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

            os.remove(output_file + ".tar")
            print(f"File '{input_file}' compressed to '{output_file}.tar.gz'.")
        
        # File not found
        else:
            print(f"Error: The file '{input_file}' does not exist.")

    except Exception as e:
        print(f"An error occurred: {e}")

def print_file_size(file_path):
    """Prints the size of the file at the given path in Mbytes."""
    CRED = '\033[91m'
    CEND = '\033[0m'
    try:
        file = Path(file_path)

        # File
        if file.is_file():
            file_size = file.stat().st_size
            file_size = file_size / 1024 / 1024
            print(f"The size of the file {file_path} is " + CRED + f"{file_size:.2f} Mb." + CEND)

        # Folder
        elif file.is_dir():
            file_size = sum(f.stat().st_size for f in file.glob('**/*') if f.is_file())
            file_size = file_size / 1024 / 1024
            print(f"The size of the file {file_path} is "  + CRED + f"{file_size:.2f} Mb." + CEND)
        
        # File not found
        else:
            print(f"Error: The file '{file_path}' does not exist.")

    except Exception as e:
        print(f"An error occurred: {e}")