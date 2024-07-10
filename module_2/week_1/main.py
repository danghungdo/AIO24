import numpy as np
import matplotlib.image as mpimg
import pandas as pd
import os
if __name__ == "__main__":
    # Ex1
    arr = np.arange(0, 10, 1)
    print(arr)
    print("-------------------")
    # Ex2
    arr21 = np.ones((3, 3)) > 0
    print(f"arr1:\n{arr21}")
    arr22 = np.ones((3, 3), dtype=bool)
    print(f"arr2:\n{arr22}")
    arr23 = np.full((3, 3), fill_value=True, dtype=bool)
    print(f"arr3:\n{arr23}")
    print("-------------------")
    # Ex3
    arr3 = np.arange(0, 10)
    print(arr3[arr3 % 2 == 1])
    print("-------------------")
    # Ex4
    arr4 = np.arange(0, 10)
    arr4[arr4 % 2 == 1] = -1
    print(arr4)
    print("-------------------")
    # Ex5
    arr5 = np.arange(10)
    arr5_2d = arr5.reshape(2, -1)
    print(arr5_2d)
    print("-------------------")
    # Ex6
    arr61 = np.arange(10).reshape(2, -1)
    arr62 = np.repeat(1, 10).reshape(2, -1)
    print(np.concatenate([arr61, arr62], axis=0))
    print("-------------------")
    # Ex7
    arr71 = np.arange(10).reshape(2, -1)
    arr72 = np.repeat(1, 10).reshape(2, -1)
    print(np.concatenate([arr71, arr72], axis=1))
    print("-------------------")
    # Ex8
    arr8 = np.array([1, 2, 3])
    print(np.repeat(arr8, 3))
    print(np.tile(arr8, 3))
    print("-------------------")
    # Ex9
    arr9 = np.array([2, 6, 1, 9, 10, 3, 27])
    index = np.nonzero((arr9 >= 5) & (arr9 <= 10))
    print(arr9[index])
    print("-------------------")

    # Ex10
    def maxx(x, y):
        if x >= y:
            return x
        else:
            return y
    a = np.array([5, 7, 9, 8, 6, 4, 5])
    b = np.array([6, 3, 4, 8, 9, 7, 1])
    pair_max = np.vectorize(maxx, otypes=[float])
    print(pair_max(a, b))
    print("-------------------")
    # Ex11
    print(np.where(a < b, b, a))
    print("-------------------")
    # Ex12
    current_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(current_dir, 'dog.jpeg')
    img = mpimg.imread(img_path)
    gray_img_01 = (np.max(img, axis=2) + np.min(img, axis=2)) / 2
    print(gray_img_01[0, 0])
    print("-------------------")
    # Ex13
    gray_img_02 = np.sum(img / 3, axis=2)
    print(gray_img_02[0, 0])
    print("-------------------")
    # Ex14
    gray_img_03 = np.dot(img[..., :3], [0.21, 0.72, 0.07])
    print(gray_img_03[0, 0])
    print("-------------------")
    # Ex15
    csv_path = os.path.join(current_dir, 'advertising.csv')
    df = pd.read_csv(csv_path)
    print(max(df.Sales), df.Sales.idxmax())
    print("-------------------")
    # Ex16
    print(df.TV.mean())
    print("-------------------")
    # Ex17
    print(df[df.Sales >= 20].shape[0])
    print("-------------------")
    # Ex18
    print(df[df.Sales >= 15].Radio.mean())
    print("-------------------")
    # Ex19
    print(df[df.Newspaper > df.Newspaper.mean()].Sales.sum())
    print("-------------------")
    # Ex20
    A = df.Sales.mean()
    scores = []
    for sale in df.Sales:
        if sale > A:
            scores.append("Good")
        elif sale == A:
            scores.append("Average")
        else:
            scores.append("Bad")
    print(scores[7:10])
    print("-------------------")
    # Ex21
    A = df.Sales[(abs(df.Sales - A)).idxmin()]
    for sale in df.Sales:
        if sale > A:
            scores.append("Good")
        elif sale == A:
            scores.append("Average")
        else:
            scores.append("Bad")
    print(scores[7:10])
    print("-------------------")
