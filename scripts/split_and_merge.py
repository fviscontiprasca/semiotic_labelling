# scripts/split_and_merge.py
import random, shutil
from pathlib import Path

random.seed(42)
OID_IMG = Path("data/oid_urban/images")
MY_IMG = Path("data/my_images")
OUT = Path("data/merged")
OUT.mkdir(parents=True, exist_ok=True)

for s in ("train","val","test"):
    (OUT/s).mkdir(parents=True, exist_ok=True)

imgs = list(OID_IMG.glob("*.jpg"))
random.shuffle(imgs)
n=len(imgs)
train = imgs[:int(0.8*n)]
val   = imgs[int(0.8*n):int(0.9*n)]
test  = imgs[int(0.9*n):]

for p in train: shutil.copy(p, OUT/"train"/p.name)
for p in val: shutil.copy(p, OUT/"val"/p.name)
for p in test: shutil.copy(p, OUT/"test"/p.name)

# add your images: distribute to keep train heavy
my_imgs = list(MY_IMG.glob("*.jpg"))
for i,img in enumerate(my_imgs):
    if i % 10 == 8:
        dst="val"
    elif i % 10 == 9:
        dst="test"
    else:
        dst="train"
    shutil.copy(img, OUT/dst/img.name)

print("Merged dataset ready at data/merged")
