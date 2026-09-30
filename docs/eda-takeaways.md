# EDA Takeaways

## Dataset

- 573,913 reviews across 1,572 movies.
- 73.7% non-spoiler and 26.3% spoiler.
- Median review length: 189 words.
- Median reviews per movie: 326.

## Data Quality

- No exact duplicate rows.
- 357 same-movie repeated-review-text groups (786 rows; 0.137%).
- 69 of these groups contain conflicting spoiler labels (181 rows).
- 76 exact review texts occur across multiple movies.
- 2 metadata `movie_id` values contain trailing `/`, causing 7 failed
  review-to-metadata matches.
- 233 movies have no plot synopsis, although every movie has a plot summary.

## Cleaning

- Normalize `movie_id` by stripping surrounding whitespace and trailing `/`.
- Do not automatically delete repeated reviews.
- Do not arbitrarily choose a target label for conflicting-label groups.
- Leave raw source files unchanged.

## Splitting Recommendations

- For text-based evaluation, all rows sharing the same exact `review_text`
  should remain in the same train/validation/test partition.
- For RQ3, all rows sharing a `movie_id` must also remain together.
- Check for exact text overlap after creating the movie-disjoint split because
  identical review text occurs across some different movies.

## Research-Question Implications

- **RQ1:** The 73.7/26.3 class imbalance should be considered when evaluating
  the text classifiers.
- **RQ2:** Plot summaries cover all movies, while plot synopses cover 85.18%;
  synopses can also be very long.
- **RQ3:** Review counts are uneven across movies, and movie-disjoint
  evaluation is necessary to test generalisation to unseen movies.
- **RQ4:** 25,699 reviews (4.48%) explicitly contain `spoiler` or `spoilers`,
  motivating the masked-vs-unmasked experiment.