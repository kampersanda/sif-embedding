import os
from glob import glob

import polars as pl
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

STOPWORDS = set(stopwords.words('english'))

# NOTE(kampersanda): Option if removing stopwords.
# I support this option because the original implementation did this process,
# but I don't see the need for this. This is because frequent words are handled
# by the smoothing term from unigram probabilities.
REMOVE_STOPWORDS = False


def clean_text(text):
    words = word_tokenize(text)
    if REMOVE_STOPWORDS:
        words = [word for word in words if word not in STOPWORDS]
    return ' '.join(words).lower()


def main():
    # nltk.download('stopwords')

    source_dir = 'semeval-sts/all'
    if REMOVE_STOPWORDS:
        target_dir = 'semeval-sts-clean-wo-stopwords/all'
    else:
        target_dir = 'semeval-sts-clean/all'

    os.makedirs(target_dir, exist_ok=False)

    source_paths = sorted(glob(f'{source_dir}/*.tsv'))
    for source_path in source_paths:
        print(source_path)
        try:
            df = pl.read_csv(
                source_path,
                has_header=False,
                separator='\t',
                new_columns=['score', 'sent1', 'sent2'],
            )
        except pl.exceptions.ComputeError as e:
            print(f'{e}, skipped')
            continue

        df = df.with_columns((pl.col('sent1').apply(lambda x: clean_text(x))))
        df = df.with_columns((pl.col('sent2').apply(lambda x: clean_text(x))))

        filename = os.path.basename(source_path)
        target_path = f'{target_dir}/{filename}'

        df.write_csv(
            target_path,
            has_header=False,
            separator='\t',
        )


if __name__ == "__main__":
    main()
