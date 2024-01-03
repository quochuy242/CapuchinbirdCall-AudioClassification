import os

print(os.getcwd())
print(os.path.join(os.getcwd(), "src", "tasks"))

os.chdir(os.path.join(os.getcwd(), "src", "tasks"))
