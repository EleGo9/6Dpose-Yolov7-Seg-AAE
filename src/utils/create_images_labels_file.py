import fileinput

file = open("dataset/images_labels.txt", "w")

file.write("images, labels\n")
for i in range(250):
    file.write(
        "{:04d}.png,{:04d}.png\n".format(i, i)
    )
file.close()
