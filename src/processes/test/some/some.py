import os
import sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.getcwd())


class FileToPlot:
    def __init__(self, file_path):
        self.file_path = file_path

        self.real_sense = []
        self.transformation = []
        self.z = []
        self.test = []

    def read_file(self):
        self.real_sense = []
        self.transformation = []
        self.test = []

        with open(self.file_path) as f:
            for l in f:
                if len(l) > 1:
                    if "real-sense" in l:
                        z = int(l.split(",")[2].strip()[:-1])
                        # print("rs: {}".format(z), end=" - ")
                        self.real_sense.append(z)
                    elif "transformation" in l:
                        z = int(l.split(",")[2].strip()[:-1])
                        # print("t: {}".format(z))
                        self.transformation.append(z)
                    elif "z" in l:
                        z = int(l.split(" ")[1].strip())
                        # print("t: {}".format(z))
                        self.z.append(z)

        '''for rs, t, z in zip(self.real_sense, self.transformation, self.z):
            print("{} - {} - {}".format(rs, t, z))'''

        self.test = range(0, len(self.real_sense))

        return self.real_sense, self.transformation, self.z, self.test


class Gap:
    def __init__(self):
        self.diff = []

    def compute(self, estimation, measure):
        for e, m in zip(estimation, measure):
            d = abs(m-e)/m
            self.diff.append(d)
            print("{} - {} ({})".format(m, e, d))
        print()

    def stats(self, estimation=None, measure=None):
        diff = np.array(self.diff)
        print(diff.mean(), diff.std())
        if estimation is not None and measure is not None:
            print(sum(estimation)/len(estimation), sum(measure)/len(measure))
            print(diff.mean()*(sum(measure)/(len(measure))), diff.mean()*(sum(estimation)/(len(estimation))))
        print()


'''small_down = FileToPlot("src/image/test/small/dado_m5/down/notes.txt")
s_d_rs, s_d_t, s_t = small_down.read_file()
small_side = FileToPlot("src/image/test/small/dado_m5/side/notes.txt")
s_s_rs, s_s_t, _ = small_side.read_file()
small_up = FileToPlot("src/image/test/small/dado_m5/up/notes.txt")
s_u_rs, s_u_t, _ = small_up.read_file()

normal_down = FileToPlot("src/image/test/normal/dado_m5/down/notes.txt")
n_d_rs, n_d_t, _ = normal_down.read_file()
normal_side = FileToPlot("src/image/test/normal/dado_m5/side/notes.txt")
n_s_rs, n_s_t, _ = normal_side.read_file()
normal_up = FileToPlot("src/image/test/normal/dado_m5/up/notes.txt")
n_u_rs, n_u_t, _ = normal_up.read_file()'''

'''figure, axis = plt.subplots(3, 1)

axis[0].plot(s_t, s_d_rs)
axis[0].plot(s_t, s_d_t)
axis[0].plot(s_t, n_d_rs, ".")
axis[0].plot(s_t, n_d_t)
axis[0].set_title("Down")

axis[1].plot(s_t, s_s_rs)
axis[1].plot(s_t, s_s_t)
axis[1].plot(s_t, n_s_rs, ".")
axis[1].plot(s_t, n_s_t)
axis[1].set_title("Side")

axis[2].plot(s_t, s_u_rs)
axis[2].plot(s_t, s_u_t)
axis[2].plot(s_t, n_u_rs, ".")
axis[2].plot(s_t, n_u_t)
axis[2].set_title("Up")'''

#closer = FileToPlot("src/image/test/closer/dado_m5/random/notes.txt")
all = FileToPlot("src/image/test/all/notes.txt")
c_d_rs, c_d_t, c_z, c_t = all.read_file()

gap = Gap()
gap.compute(c_d_t, c_d_rs)
gap.stats(c_d_t, c_d_rs)

plt.ylim(0, 3700)
plt.plot(c_t, c_d_rs, "--")
plt.plot(c_t, c_d_t)
#plt.plot(c_t, c_z)
#plt.plot(c_t, [550 for i in c_t], "--")
plt.show()
