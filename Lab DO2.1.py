import os
import subprocess

def run_command(command):
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    return result.returncode

project_name = "calculator_project"
os.makedirs(project_name, exist_ok=True)
os.chdir(project_name)
print(f"Рабочая директория создана: {os.getcwd()}")

run_command("git init")

run_command("git status")

file_name = "calculator.py"
with open(file_name, "w") as f:
    f.write("# Simple Calculator\n")
print(f"Файл {file_name} создан.")

run_command("git status")

run_command(f"git add {file_name}")
run_command("git status")

run_command('git commit -m "Initial commit: added calculator.py"')

with open(file_name, "a") as f:
    f.write("""
def add(a, b):
    return a + b
""")
print(f"Функция add добавлена в {file_name}.")
run_command("git commit -am 'Added addition function to calculator.py'")

with open(file_name, "a") as f:
    f.write("""
def subtract(a, b):
    return a - b
""")
print(f"Функция subtract добавлена в {file_name}.")
run_command("git commit -am 'Added subtraction function to calculator.py'")

run_command("git log")

additional_files = ["multiply.py", "divide.py"]
for fname in additional_files:
    with open(fname, "w") as f:
        f.write("# This is an empty file\n")
    print(f"Файл {fname} создан.")
run_command("git add .")
run_command('git commit -m "Added multiply.py and divide.py"')

run_command("git branch feature/multiplication")
run_command("git checkout feature/multiplication")

with open("multiply.py", "w") as f:
    f.write("""
def multiply(a, b):
    return a * b
""")
print("Функция multiply добавлена в ветке feature/multiplication.")
run_command('git add multiply.py')
run_command('git commit -m "Added multiply function in feature/multiplication branch"')

run_command("git checkout main")
run_command("git merge feature/multiplication")

run_command('git tag -a v1.0 -m "Release version 1.0"')
run_command("git tag")

with open(".gitignore", "w") as f:
    f.write("venv/\n*.pyc\n__pycache__/\n")
print(".gitignore файл создан.")
run_command("git add .gitignore")
run_command('git commit -m "Added .gitignore for virtual environment"')

run_command("git branch")