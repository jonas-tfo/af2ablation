import os

import tqdm

for i in tqdm.tqdm(
    """A2RJ53
O76728
P31133
P00558
P40131
Q7DAU8
P21589
A0QTT2
Q5F9M1
Q9X6R4
Q18A65
Q9ERE7
P62495
A0A075Q0W3
P71447
P33284
Q9Z4N6
A6UVT1
B7IE18
B3EYN2
Q53W80
Q9SS90
Q9X9P9""".split("\n")
):
    os.system(
        f"python -m afpert.scripts.run_predictions --depths 32,256,1024,5120 --method custom --target-dir {i}"
    )
