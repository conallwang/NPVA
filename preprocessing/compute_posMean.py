import os
import numpy as np
from tqdm import tqdm
posmap_root = "/path/to/datasets/multiface/m--20190529--1300--002421669--GHS/gendir/posMap"
output_dir = "/path/to/datasets/multiface/m--20190529--1300--002421669--GHS/gendir/"
print("Compute posMean ...")
count = 0
posMean = 0
bar = tqdm(range(len(os.listdir(posmap_root))))  # 185 expressions
for expr in os.listdir(posmap_root):
    expr_dir = os.path.join(posmap_root, expr)
    for file in os.listdir(expr_dir):
        if file[-4:] != ".npy":
            continue
        posMap = np.load(os.path.join(expr_dir, file))  # [256, 256, 3]
        posMean = (count / (count + 1)) * posMean + (posMap / (count + 1))
        count += 1
    bar.update()
posMean_path = output_dir + "/posMean.npy"
np.save(posMean_path, posMean.transpose((2, 0, 1)))
print("posMean has been saved: " + posMean_path)
# posMean = np.load(posMean_path).transpose((1, 2, 0))
print("\nCompute posVar ...")
count = 0
posVar = 0
bar = tqdm(range(len(os.listdir(posmap_root))))  # 185 expressions
for expr in os.listdir(posmap_root):
    expr_dir = os.path.join(posmap_root, expr)
    for file in os.listdir(expr_dir):
        if file[-4:] != ".npy":
            continue
        posMap = np.load(os.path.join(expr_dir, file))  # [256, 256, 3]
        var = (posMap - posMean) ** 2
        posVar = (count / (count + 1)) * posVar + (var / (count + 1))  # [256, 256, 3]
        count += 1
    bar.update()
posVar = posVar.mean()
posVar_path = output_dir + "/posVar.txt"
np.savetxt(posVar_path, [posVar])
print("posStd has been saved: " + posVar_path)