import sys

data = sys.stdin.read()
data = data.split("\n")

packages = dict()
for line in data:
    if len(line.split("==")) > 1:
        packages[line.split("==")[0]] = line.split("==")[1]

new_requirements = []
with open("./requirements.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        package = line.split("==")[0]
        if package[-1] == "\n":
            package = package[:-1]
        new_requirements.append(package + "==" + packages[package])

with open("./requirements.txt", "w") as f:
    f.write("\n".join(new_requirements))
