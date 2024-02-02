import os

directory_path = "src/results/150923/single_object/real_lessaae/{}"
directory = ["chiave", "dado", "ugello", "vite"]


for d in directory:
    files = os.listdir(directory_path.format(d))
    prefix = {}
    for file in files:
        p = file.split("_")[0] + "_" + file.split("_")[1]
        if p not in prefix.keys():
            prefix[p] = 1
        else:
            prefix[p] += 1

    print(len(prefix), prefix)
    print()
