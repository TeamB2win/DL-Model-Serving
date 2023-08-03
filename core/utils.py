import os


def clear_working_dir():
  print("Getting clear working directory")
  working_dir = os.environ['WORKING_DIR']
  filenames = os.listdir(working_dir)
  
  for filename in filenames:
    file_path = os.path.join(working_dir, filename)
    os.remove(file_path)

    print(f"Remove {file_path}")