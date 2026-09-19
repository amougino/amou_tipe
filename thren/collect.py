import pandas as pd


def otf_collect_single_l(folder, variation, constants_condition=[]):
    '''
    constants_condition : 
        list of 2-uples with the name and value
        ex : [("sat_vel_y", 200)]
            returns the variation of the desired value where "sat_vel_y" is stable at 200
    '''
    df = pd.read_csv(folder + "/data.csv")
    for constant in constants_condition:
        df = df[df[constant[0]] == constant[1]]
    return (
        df[variation].tolist(),
        df["v"].tolist(),
        df["e"].tolist(),
        df["theta"].tolist(),
        df["min_b1"].tolist(),
        df["min_b2"].tolist()
    )


def otf_collect_multi_l(folder, variations):
    df = pd.read_csv(folder + "/data.csv")
    var_lists = [df[col].tolist() for col in variations]
    other_lists = [df[col].tolist() for col in ["v", "e", "theta", "min_b1", "min_b2"]]
    return var_lists, other_lists


def otf_collect_multi_df(folder, columns):
    df = pd.read_csv(folder + "/data.csv")
    return df[columns]
