import thren.sim as sim
import thren.parallel_otf as parallel_otf


def main():
    folder = parallel_otf.parallel_otf(
        sim.__location__ + "/settings_complete.json",
        workers="80%",
        chunksize=256,
    )


if __name__ == "__main__":
    main()
