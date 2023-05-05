import os
from glob import glob

import polars as pl
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

STOPWORDS = set(stopwords.words('english'))


def clean_text(text):
    words = word_tokenize(text)
    words = [word for word in words if word not in STOPWORDS]
    return ' '.join(words).lower()


def main():
    # nltk.download('stopwords')

    source_dir = 'semeval-sts/all'
    target_dir = 'semeval-sts-clean/all'

    os.makedirs(target_dir, exist_ok=True)

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
