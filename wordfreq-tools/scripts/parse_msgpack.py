import argparse
import gzip

import msgpack


def cB_to_freq(cB: int) -> float:
    """
    Convert a word frequency from the logarithmic centibel scale that we use
    internally, to a proportion from 0 to 1.

    On this scale, 0 cB represents the maximum possible frequency of
    1.0. -100 cB represents a word that happens 1 in 10 times,
    -200 cB represents something that happens 1 in 100 times, and so on.

    In general, x cB represents a frequency of 10 ** (x/100).
    """
    if cB > 0:
        raise ValueError("A frequency cannot be a positive number of centibels.")
    return 10 ** (cB / 100)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str)
    args = parser.parse_args()

    with gzip.open(args.input, "rb") as infile:
        data = msgpack.load(infile, raw=False)

    header = data[0]
    if header != {'format': 'cB', 'version': 1}:
        raise ValueError(f"Unexpected header: {header}")

    pack = data[1:]
    for index, bucket in enumerate(pack):
        freq = cB_to_freq(-index)
        for word in bucket:
            print(word, freq)


if __name__ == "__main__":
    main()
