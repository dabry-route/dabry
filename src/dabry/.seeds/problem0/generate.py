import csv
import random

with open('center1.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows([[0.5, 0.8]] + [[-0.2 + random.random()*1.4, -1. + random.random() * 2.] for _ in range(999)])

with open('center2.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows([[0.8, 0.2]] + [[-0.2 + random.random()*1.4, -1. + random.random() * 2.] for _ in range(999)])

with open('center3.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows([[0.6, -0.5]] + [[-0.2 + random.random() * 1.4, -1. + random.random() * 2.] for _ in range(999)])
