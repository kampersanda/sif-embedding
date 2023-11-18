import argparse


def load_scores(path):
    text = open(path, "r").read()
    chunks = text.split("\n\n")
    scores = {}
    for chunk in chunks:
        lines = chunk.split("\n")
        for line in lines[1:]:
            dir, file, pearson, spearman = line.split("\t")
            if dir not in scores:
                scores[dir] = {}
            scores[dir][file] = (pearson, spearman)
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sif_glove")
    parser.add_argument("sif_fasttext")
    parser.add_argument("usif_glove")
    parser.add_argument("usif_fasttext")
    args = parser.parse_args()

    sif_glove_scores = load_scores(args.sif_glove)
    sif_fasttext_scores = load_scores(args.sif_fasttext)
    usif_glove_scores = load_scores(args.usif_glove)
    usif_fasttext_scores = load_scores(args.usif_fasttext)

    dirs = [
        "STS/STS12-en-test",
        "STS/STS13-en-test",
        "STS/STS14-en-test",
        "STS/STS15-en-test",
        "STS/STS16-en-test",
        "STS/STSBenchmark",
        "SICK",
    ]

    print("[Pearson]\n")
    for dir in dirs:
        sif_glove_score = sif_glove_scores[dir]
        sif_fasttext_score = sif_fasttext_scores[dir]
        usif_glove_score = usif_glove_scores[dir]
        usif_fasttext_score = usif_fasttext_scores[dir]

        print(
            "\t".join(
                [
                    dir,
                    "Sif<br>(GloVe)",
                    "Sif<br>(fastText)",
                    "USif<br>(GloVe)",
                    "USif<br>(fastText)",
                ]
            )
        )
        for file in sif_glove_score:
            sif_glove_pearson, _ = sif_glove_score[file]
            sif_fasttext_pearson, _ = sif_fasttext_score[file]
            usif_glove_pearson, _ = usif_glove_score[file]
            usif_fasttext_pearson, _ = usif_fasttext_score[file]
            print(
                "\t".join(
                    [
                        file,
                        sif_glove_pearson,
                        sif_fasttext_pearson,
                        usif_glove_pearson,
                        usif_fasttext_pearson,
                    ]
                )
            )
        print()

    print("[Spearman]\n")
    for dir in dirs:
        sif_glove_score = sif_glove_scores[dir]
        sif_fasttext_score = sif_fasttext_scores[dir]
        usif_glove_score = usif_glove_scores[dir]
        usif_fasttext_score = usif_fasttext_scores[dir]

        print(
            "\t".join(
                [
                    dir,
                    "Sif<br>(GloVe)",
                    "Sif<br>(fastText)",
                    "USif<br>(GloVe)",
                    "USif<br>(fastText)",
                ]
            )
        )
        for file in sif_glove_score:
            _, sif_glove_spearman = sif_glove_score[file]
            _, sif_fasttext_spearman = sif_fasttext_score[file]
            _, usif_glove_spearman = usif_glove_score[file]
            _, usif_fasttext_spearman = usif_fasttext_score[file]
            print(
                "\t".join(
                    [
                        file,
                        sif_glove_spearman,
                        sif_fasttext_spearman,
                        usif_glove_spearman,
                        usif_fasttext_spearman,
                    ]
                )
            )
        print()


if __name__ == "__main__":
    main()
