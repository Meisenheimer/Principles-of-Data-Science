import random
import numpy as np
from PIL import Image
from matplotlib import pyplot

img0 = Image.open("0.jpg").crop((265, 106, 2933, 2774)).resize((1024, 1024))
img1 = Image.open("1.jpg").crop((265, 106, 2933, 2774)).resize((1024, 1024))
img2 = Image.open("2.jpg").crop((265, 106, 2933, 2774)).resize((1024, 1024))
img3 = Image.open("3.jpg").crop((265, 106, 2933, 2774)).resize((1024, 1024))
img4 = Image.open("4.jpg").crop((265, 106, 2933, 2774)).resize((1024, 1024))

img = Image.new("RGB", (1024 + 4 * 256, 1024 + 4 * 256), color=(255, 255, 255))
img.paste(img0, (0 * 256, 0 * 256))
img.paste(img1, (1 * 256, 1 * 256))
img.paste(img2, (2 * 256, 2 * 256))
img.paste(img3, (3 * 256, 3 * 256))
img.paste(img4, (4 * 256, 4 * 256))
img.save("input.jpg")
img0.save("0_crop.jpg")
img1.save("1_crop.jpg")
img2.save("2_crop.jpg")
img3.save("3_crop.jpg")
img4.save("4_crop.jpg")

N = 1000

random.seed(1)
np.random.seed(1)

pyplot.clf()
pyplot.figure(figsize=(12, 2))
pyplot.grid()
pyplot.xticks(ticks=[], labels=[])
pyplot.yticks(ticks=[], labels=[])
pyplot.plot(range(N), np.random.randn(N), "k-")
pyplot.savefig("ts.jpg", dpi=720, bbox_inches="tight")
