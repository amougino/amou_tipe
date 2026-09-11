import PIL.Image as Img
import os
import collect

brgbw = {
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
    "end": (1, 1, 1)
}


def valid(folder, two_vars):
    (v1, v2), (multi_v, multi_e, multi_theta, multi_min_b1, multi_min_b2) = collect.otf_collect_multi(folder, two_vars)

# TO DO : test this thingy i dont trust it at all
# then try to create an image and a matplotlib plot from the data


def color_code(code, value, mini=0, maxi=1):
    new_val = value - mini
    new_max = maxi - mini

    if new_val == 0:
        return code["start"]
    elif new_val == new_max:
        return code["end"]
    else:
        n_patterns = len(code["pattern"])
        ratio = new_val/new_max
        for i in range(1, n_patterns):
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
        raise Exception("unable to match")


def createImg(size):
    img = Img.new('RGB', size)
    return img


def saveImg(img, path):
    file = os.path.expanduser(path)
    img.save(file)
