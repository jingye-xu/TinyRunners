import os
import re

import numpy as np
import pandas as pd

VA = [
    "VT",
    "VFb",
    "VFt"
    ]

NON_VA = [
    "AFb",
    "AFt",
    "SR",
    "SVT",
    "VPD",
    ]


def read_txt(file_path:str)-> np.ndarray:
    with open(file_path) as f:
        lines = f.readlines()
        data = np.array(lines, dtype=np.float32)
        return data


data_list = []
for root, dirs, files in os.walk("data/tinyml_contest_data_training/"):
    for name in files:
        path = os.path.join(root, name)

        data_array = read_txt(path)

        # all files has 1250 length
        if data_array.shape[0] != 1250:
            print(path)
        
        result = re.findall("(.*)-(.*)-(.*).txt", name)
        result = result[0]
        data = {
            "subject": result[0],
            "rhythm": result[1],
            "segment_index": result[2],
            "full_path": path,
            "file_name": name,
            "is_va": result[1] in VA,
            "data": data_array
        }
        
        data_list.append(data)


df=pd.DataFrame(data_list)

df.to_pickle("data/data.gz")

