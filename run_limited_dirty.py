import os

import tqdm

for i in tqdm.tqdm(
"""A2RJ53
P31133
P40131
Q7DAU8
P00558
A0QTT2
Q5F9M1
Q18A65
Q9ERE7
P62495
P71447
Q9Z4N6
A6UVT1
Q53W80
Q9SS90
Q9X9P9""".split("\n")
):
    os.system(
        f"python -m afpert.scripts.run_predictions --depths 32,256,1024,5120 --method afsample2 --target-dir {i}"
    )
