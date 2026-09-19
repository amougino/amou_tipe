import PIL.Image as Img
import os
import collect
import numpy as np

BRGBW = {  # Black Red Green Blue White
    "start": (0, 0, 0),
    "pattern": [
        ["a", 0, 0],
        [1, "a", 0],
        ["d", 1, 0],
        [0, 1, "a"],
        [0, "d", 1],
        ["a", 0, 1],
        [1, "a", 1]
    ],
    "end": (1, 1, 1),
    "nan": (.5, .5, .5)
}

'''
is this actually useful??


def valid_for_mapping(folder, two_vars):
    (v1, v2), (multi_v, multi_e, multi_theta, multi_min_b1, multi_min_b2) = collect.otf_collect_multi(folder, two_vars)
    print(multi_theta)
'''


def create_img(size):
    img = Img.new('RGB', size)
    return img


def save_img(img, path):
    file = os.path.expanduser(path)
    img.save(file)


def color_code(code, value, mini=0, maxi=1):
    if np.isnan(value):
        return tuple(int(i*255) for i in code["nan"])
    else:
        new_val = value - mini
        new_max = maxi - mini

        if new_val == 0:
            return tuple(int(i*255) for i in code["start"])
        elif new_val == new_max:
            return tuple(int(i*255) for i in code["end"])
        else:
            n_patterns = len(code["pattern"])
            ratio = new_val/new_max
            for i in range(1, n_patterns+1):
                if ratio < i/n_patterns:
                    rgb = []
                    for j in range(3):
                        if code["pattern"][i - 1][j] == "a":
                            rgb.append(int((ratio*n_patterns*255)-(255*(i-1))))
                        elif code["pattern"][i - 1][j] == "d":
                            rgb.append(int((255*i)-(ratio*n_patterns*255)))
                        else:
                            rgb.append(255*code["pattern"][i - 1][j])
                    return tuple(rgb)
            print(value, mini, maxi, ratio)
            raise Exception("unable to match")


def mapping(folder, two_vars, plotted_value, type="png", colors=BRGBW, save_folder=None):
    '''
    folder : data to plot
    two_vars : variables for x and y axis in image
    plotted_value : the value is represented
    type : currently NULL
    colors : the color code
    '''
    df = collect.otf_collect_multi_df(folder, two_vars + [plotted_value])
    to_plot = df[plotted_value].tolist()
    mini = min(to_plot)
    maxi = max(to_plot)
    v1_set = sorted(set(df[two_vars[0]].tolist()))
    n1 = len(v1_set)
    v2_set = sorted(set(df[two_vars[1]].tolist()))
    n2 = len(v2_set)

    if type == "png":
        data_map = create_img((n1, n2))
        for idx1, v1 in enumerate(v1_set):
            for idx2, v2 in enumerate(v2_set):
                value = df[(df[two_vars[0]] == v1) & (df[two_vars[1]] == v2)][plotted_value].iloc[0]
                print(value)
                data_map.putpixel((idx1, idx2), color_code(colors, value, mini=mini, maxi=maxi))
        data_map.show()
        if save_folder != None:
            save_img(
                data_map,
                save_folder + "/" + "_".join([
                    "img",
                    plotted_value,
                    str(n1)+"x"+str(n2),
                    two_vars[0],
                    str(v1_set[0]),
                    str(v1_set[-1]),
                    two_vars[1],
                    str(v2_set[0]),
                    str(v2_set[-1]),
                    ".png"
                ])
            )

# TO DO : try to create an image and a matplotlib plot from the data
