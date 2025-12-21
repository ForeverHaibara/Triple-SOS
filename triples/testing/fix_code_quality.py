import os
import shutil
def fix_code_quality(path):
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                except Exception:
                    print(f"Error reading {file_path}")
                    continue
                new_lines = []
                for line in lines:
                    new_lines.append(line.rstrip() + "\n")
                # remove trailing blank lines
                while new_lines and new_lines[-1] == "\n":
                    new_lines.pop()
                with open(file_path, "w", encoding="utf-8") as f:
                    f.writelines(new_lines)


def remove_pycache(path):
    for root, dirs, files in os.walk(path):
        for dir in dirs:
            if dir == "__pycache__" or dir == ".pytest_cache":
                dir_path = os.path.join(root, dir)
                shutil.rmtree(dir_path)
                # print(f"Removed {dir_path}")


def fix_code_quality_main():
    path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print("Directory =", path)
    print("Select an option:")
    print("1. Remove trailing whitespace & add blank lines")
    print("2. Remove __pycache__ and .pytest_cache directories")

    option = input("Enter your choice: ")
    if option == "1":
        print(f"This will remove trailing whitespace & add blank lines to all files in {path}.")
        func = fix_code_quality
    elif option == "2":
        print(f"This will remove all __pycache__ and .pytest_cache directories in {path}.")
        func = remove_pycache
    else:
        print("Aborted.")
        return
    confirm = input("Proceed? (y/n)")
    if confirm == "y":
        func(path)
    else:
        print("Aborted.")

if __name__ == '__main__':
    fix_code_quality_main()
