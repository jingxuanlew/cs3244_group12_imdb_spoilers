"""Remove or mask explicit spoiler cues in the IMDb reviews JSON-lines file."""

import argparse
from datetime import date
import json
from pathlib import Path
import re


# Edit this list to change the ablation. Keep ordinary plot vocabulary intact.
SPOILER_TERMS = [
    "spoiler", "spoilers",
    "spoiler alert", "spoilers alert",
    "spoiler warning", "spoilers warning",
    "warning spoiler", "warning spoilers",
    "spoiler ahead", "spoilers ahead",
    "contains spoiler", "contains spoilers",
    "contain spoilers", "may contain spoilers",
    "possible spoiler", "possible spoilers",
    "minor spoiler", "minor spoilers",
    "major spoiler", "major spoilers",
    "no spoiler", "no spoilers", "without spoilers",
    "spoiler free", "non spoiler", "non spoilers",
    "end spoiler", "end spoilers",
]
TEXT_FIELDS = ("review_text", "review_summary")


SPOILER_PATTERN = re.compile(
    r"\b(?:"
    + "|".join(
        re.escape(term).replace(r"\ ", r"[\s_-]+")
        for term in sorted(SPOILER_TERMS, key=len, reverse=True)
    )
    + r")\b",
    flags=re.IGNORECASE,
)


def clean_text(text, replacement=" "):
    """Return cleaned text. Preserve missing values and text without matches."""
    if not isinstance(text, str):
        return text
    cleaned, matches = SPOILER_PATTERN.subn(replacement, text)
    if matches:
        return re.sub(r"[ \t]+", " ", cleaned).strip()
    return text


def clean_reviews(input_path, output_path):
    """Stream reviews to a separate file, returning total and changed row counts."""
    input_path, output_path = Path(input_path), Path(output_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output must be different files.")

    total = changed = 0
    with input_path.open(encoding="utf-8") as source:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as target:
            for line in source:
                if not line.strip():
                    continue
                review = json.loads(line)
                row_changed = False
                for field in TEXT_FIELDS:
                    if field in review:
                        original = review[field]
                        review[field] = clean_text(original, " ")
                        row_changed |= review[field] != original
                target.write(json.dumps(review, ensure_ascii=False) + "\n")
                total += 1
                changed += row_changed
    return total, changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/IMDB_reviews.json"), help="Input JSON-lines file (one review per line).")
    parser.add_argument("--output", type=Path, help="Output file; defaults to data/processed/IMDB_reviews_<mode>.json.")
    args = parser.parse_args()
    output = args.output or Path(f"data/processed/IMDB_reviews_{date.today()}.json")
    total, changed = clean_reviews(args.input, output)
    print(f"Processed {total} reviews, changed {changed} reviews.")
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
