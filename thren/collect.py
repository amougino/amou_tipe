import pandas as pd


def otf_collect(folder, variation, constants):
    df = pd.read_csv(folder + "/data.csv")
    for constant in constants:
        df = df[df[constant[0]] == constant[1]]
    return df[variation], df["v"], df["e"], df["theta"], df["min_b1"], df["min_b2"]
