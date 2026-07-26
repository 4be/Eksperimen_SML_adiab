"""Otomatisasi preprocessing Wine Recognition Dataset."""

from pathlib import Path

import pandas as pd


def preprocess_data(input_path: str | Path, output_path: str | Path | None = None):
    """Memuat, membersihkan, dan mengembalikan dataset siap latih."""
    dataframe = pd.read_csv(input_path)
    processed = dataframe.drop_duplicates().reset_index(drop=True)
    feature_columns = [
        column for column in processed.columns if column != "target"
    ]

    for column in feature_columns:
        processed[column] = pd.to_numeric(processed[column], errors="coerce")
        processed[column] = processed[column].fillna(processed[column].median())
        lower_limit = processed[column].quantile(0.01)
        upper_limit = processed[column].quantile(0.99)
        processed[column] = processed[column].clip(lower_limit, upper_limit)

    processed["target"] = pd.to_numeric(
        processed["target"], errors="raise"
    ).astype(int)

    if output_path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        processed.to_csv(output, index=False)

    return processed


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    raw_path = project_root / "wine_raw.csv"
    processed_path = Path(__file__).resolve().parent / "wine_preprocessing.csv"
    result = preprocess_data(raw_path, processed_path)
    print(f"Preprocessing selesai: {result.shape}")
    print(f"Output: {processed_path}")
