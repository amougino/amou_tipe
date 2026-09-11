import pandas as pd


def otf_collect_single(folder, variation, constants_condition):
    df = pd.read_csv(folder + "/data.csv")
    for constant in constants_condition:
        df = df[df[constant[0]] == constant[1]]
    return df[variation], df["v"], df["e"], df["theta"], df["min_b1"], df["min_b2"]


def otf_collect_multi(folder, variations):
    df = pd.read_csv(folder + "/data.csv")
    return df[variations], df[["v", "e", "theta", "min_b1", "min_b2"]]
