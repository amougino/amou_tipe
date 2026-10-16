import PIL.Image as Img
import os
import json
import thren.collect as collect
import thren.formulae as formulae
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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
    "nan": (.75, .75, .75)
}


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


class Mapper:

    def __init__(self, folder, two_vars, quantity):
        self.folder = folder
        self.var1 = two_vars[0]
        self.var2 = two_vars[1]
        self.quantity = quantity

        self.collect()

    def collect(self):
        df = collect.otf_collect_multi_df(
            self.folder,
            [self.var1, self.var2, self.quantity]
        )
        table = df.pivot_table(
            index=self.var1,
            columns=self.var2,
            values=self.quantity,
            aggfunc="first"
        )

        self.v1_set = table.index.tolist()
        self.v2_set = table.columns.tolist()
        self.size = (len(self.v1_set), len(self.v2_set))

        self.data = table.to_numpy(dtype=float)
        self.mini = np.nanmin(self.data)
        self.maxi = np.nanmax(self.data)

    def png_name(self):
        return "_".join([
            "img",
            self.quantity,
            "x".join(map(str, self.size)),
            self.var1,
            str(self.v1_set[0]),
            str(self.v1_set[-1]),
            self.var2,
            str(self.v2_set[0]),
            str(self.v2_set[-1])
        ]) + ".png"

    def to_png(self, color_map=BRGBW, save_folder=None):

        img = create_img(self.size)

        for i in range(self.size[0]):
            for j in range(self.size[1]):
                img.putpixel((i, j), color_code(color_map, self.data[i][j], mini=self.mini, maxi=self.maxi))

        img.show()
        if save_folder != None:
            save_img(
                img,
                save_folder + "/" + self.png_name()
            )

    def to_plt(self, color_map="viridis", ax1=None, ax2=None, title=None, phases=None, constants=None):

        fig, ax = plt.subplots()

        x0, x1 = self.v1_set[0], self.v1_set[-1]
        y0, y1 = self.v2_set[0], self.v2_set[-1]

        im = ax.imshow(
            self.data.T,
            cmap=color_map,
            origin="lower",
            extent=[x0, x1, y0, y1],
            aspect="auto"
        )
        fig.colorbar(im, ax=ax)

        ax.set_xlabel(self.var1 if ax1 is None else ax1)
        ax.set_ylabel(self.var2 if ax2 is None else ax2)
        ax.set_title(self.quantity if title is None else title)

        if phases is not None:
            self.phase_lines(ax, phases, constants)

        plt.show()

    def phase_lines(self, ax, phases=(0.0,), constants=None, resolution=400, max_lines=60):
        '''
        draws, on an existing plot, the lines along which the arrival phase is the same

        phases    : phase(s) in radians, one color per phase (ex: 0, or [0, np.pi])
        constants : settings that are not varied (mass1, mass2, body_distance, ...),
                    read from constants.json in the results folder if None
        max_lines : cap on the number of lines per phase, they pile up near the escape
                    limit v0^2 = 2*G*m/x0 where the arrival time tends to infinity
        '''
        if constants is None:
            path = os.path.join(self.folder, "constants.json")
            if not os.path.isfile(path):
                raise FileNotFoundError(f"{path} not found, pass constants=<settings dict>")
            with open(path) as f:
                constants = json.load(f)

        phases = np.atleast_1d(np.asarray(phases, dtype=float))
        xlim, ylim = ax.get_xlim(), ax.get_ylim()

        # fine grid over what is currently displayed, independent of the data grid
        X, Y = np.meshgrid(
            np.linspace(*xlim, resolution),
            np.linspace(*ylim, resolution)
        )
        settings = dict(constants)
        settings[self.var1] = X
        settings[self.var2] = Y
        total = formulae.arrival_phase_unwrapped(settings)

        finite = total[np.isfinite(total)]
        if finite.size == 0:
            print("phase_lines: no unbound trajectory in this area, nothing to draw")
            return

        total = np.where(np.isfinite(total), total, np.nan)
        lo, hi = finite.min(), finite.max()
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        handles = []

        for n, phase in enumerate(phases):
            # the same phase every 2*pi of total phase
            k_min = int(np.ceil((lo - phase) / (2 * np.pi)))
            k_max = int(np.floor((hi - phase) / (2 * np.pi)))
            levels = phase + 2 * np.pi * np.arange(k_min, k_max + 1)
            if len(levels) == 0:
                continue
            if len(levels) > max_lines:
                print(f"phase_lines: {len(levels)} lines for phase {phase:.3f}, keeping the first {max_lines}")
                levels = levels[:max_lines]
            color = color_cycle[n % len(color_cycle)]
            ax.contour(X, Y, np.ma.masked_invalid(total), levels=levels,
                       colors=[color], linewidths=1.5, linestyles="solid")
            handles.append(Line2D([0], [0], color=color, lw=1.2, label=f"arrival phase = {phase:.2f} rad"))

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        if handles:
            ax.legend(handles=handles, loc="upper right", fontsize=8)
