# GitHub Code Review Dataset Analysis

## Dataset Overview

| Split | Rows | File Size |
|-------|------|-----------|
| Train | 334,323 | 556 MB |
| Validation | 10,471 | 18 MB |
| Test | 11,013 | 19 MB |

**Total**: 355,807 code review examples

## Schema

| Column | Type | Description |
|--------|------|-------------|
| before_code | String | Code before the review |
| after_code | String | Code after applying the review |
| reviewer_comment | String | The review comment |
| diff_context | String | Diff context around the change |
| file_path | String | File path in the repository |
| comment_line | Int32 | Line number of the comment |
| language | String | Programming language |
| quality_score | Float32 | Quality score (0.07 - 1.0) |
| comment_type | String | Category of the comment |
| comment_length | Int32 | Length of the comment in characters |
| before_lines | Int32 | Number of lines before |
| after_lines | Int32 | Number of lines after |
| is_negative | Boolean | Whether the comment is negative |
| pr_title | String | Pull request title |
| pr_number | Int32 | Pull request number |
| repo_name | String | Repository name |
| repo_stars | Int64 | Repository star count |
| repo_language | String | Primary repo language |
| reviewer_username | String | Reviewer username |
| author_username | String | PR author username |

## Language Distribution (Train Split - 50k sample)

| Language | Count | Percentage |
|----------|-------|------------|
| Python | 10,775 | 21.6% |
| TypeScript | 8,370 | 16.7% |
| C++ | 6,293 | 12.6% |
| Rust | 4,469 | 8.9% |
| JavaScript | 4,035 | 8.1% |
| C# | 2,795 | 5.6% |
| C/C++ | 2,640 | 5.3% |
| Go | 2,276 | 4.6% |
| Java | 1,959 | 3.9% |
| Kotlin | 1,631 | 3.3% |

**Total languages**: 29 (including CUDA, Elixir, SQL, GraphQL, Protocol Buffers)

## Comment Type Distribution

| Type | Count | Percentage | Avg Quality Score |
|------|-------|------------|-------------------|
| suggestion | 23,392 | 46.8% | 0.57 |
| none | 10,916 | 21.8% | 1.00 |
| question | 6,578 | 13.2% | 0.40 |
| bug | 3,576 | 7.2% | 0.73 |
| refactor | 1,958 | 3.9% | 0.60 |
| performance | 1,222 | 2.4% | 0.67 |
| nitpick | 832 | 1.7% | 0.41 |
| security | 808 | 1.6% | 0.72 |
| style | 718 | 1.4% | 0.47 |

**Key Insights**:
- "none" type has perfect quality score (1.0) - likely auto-approved reviews
- Bug and security comments have highest quality scores (~0.72-0.73)
- Questions and nitpicks have lowest quality scores (~0.40)
- Suggestions dominate at ~47% of all comments

## Quality Score Analysis

| Statistic | Value |
|-----------|-------|
| Mean | 0.655 |
| Std Dev | 0.285 |
| Min | 0.071 |
| 25% | 0.429 |
| 50% (Median) | 0.643 |
| 75% | 1.000 |
| Max | 1.000 |

## Sentiment Distribution

| is_negative | Count | Percentage |
|-------------|-------|------------|
| False | 39,084 | 78.2% |
| True | 10,916 | 21.8% |

## Comment Length Statistics

| Statistic | Value (chars) |
|-----------|---------------|
| Mean | 180 |
| Std Dev | 374 |
| Min | 16 |
| 25% | 38 |
| 50% (Median) | 99 |
| 75% | 212 |
| Max | 25,500 |

**Note**: High std dev and max suggest some very long comments (possibly with code blocks)

## Code Size Statistics

| Statistic | Before Lines | After Lines |
|-----------|--------------|-------------|
| Mean | 64 | 59 |
| Std Dev | 248 | 183 |
| Min | 2 | 2 |
| Median | 51 | 51 |
| Max | 14,595 | 12,104 |

**Note**: The median of 51 lines with high variance suggests many reviews are on focused changes, but some involve large refactors.

## Repository Statistics

| Statistic | Stars |
|-----------|-------|
| Mean | 47,760 |
| Std Dev | 66,849 |
| Min | 321 |
| Median | 28,127 |
| Max | 418,799 |

### Top 10 Repositories by Review Count

| Repository | Reviews |
|------------|---------|
| bitcoin/bitcoin | 3,520 |
| dragonflydb/dragonfly | 2,281 |
| ManimCommunity/manim | 1,942 |
| facebook/react | 1,495 |
| carbon-language/carbon-lang | 1,234 |
| bevyengine/bevy | 1,082 |
| JuliaLang/julia | 1,010 |
| PowerShell/PowerShell | 930 |
| freeCodeCamp/freeCodeCamp | 908 |
| Avaiga/taipy | 903 |

## Top Reviewers

| Username | Reviews |
|----------|---------|
| (empty - auto-approval) | 10,916 |
| Copilot | 4,828 |
| duburcqa | 540 |
| vitaut | 493 |
| l0rinc | 493 |
| iSazonov | 467 |
| behackl | 437 |
| romange | 429 |
| maflcko | 341 |
| jonmeow | 328 |

**Note**: GitHub Copilot is the 2nd most active "reviewer", suggesting AI-assisted reviews are common.

## Split Consistency

All three splits (train/validation/test) show similar distributions:
- Python is consistently the top language across all splits
- Suggestion is the dominant comment type
- Language and comment type distributions are well-balanced

## Sample Reviewer Comments

1. "Should we add `if (this.owner.getX() === this.owner.getDrawableX())` to avoid extra computations 90% of the time at the cost of the `if`?"
2. Suggestion code blocks with markdown formatting
3. Explanatory comments about code changes
4. Deprecation guidance
5. Alternative API recommendations

## Key Findings for Fine-tuning

1. **Imbalanced comment types**: Suggestions dominate (~47%). Consider stratified sampling or weighting.

2. **Quality score as filter**: Use `quality_score > 0.5` to filter low-quality reviews.

3. **Language-specific models**: Python and TypeScript together account for ~38% of data. Consider language-specific fine-tuning.

4. **Negative sentiment is minority**: Only ~22% are negative. May need oversampling for sentiment-aware models.

5. **Copilot reviews**: ~10% of reviews are from GitHub Copilot. Consider whether to include/exclude based on use case.

6. **Code context**: `before_code`, `after_code`, and `diff_context` provide rich context for code review generation tasks.

7. **Repository quality**: All repos have 300+ stars, ensuring code quality baseline.

---

## Training a Small Qwen Model for Python Code Review

### Recommended Fields for Training

| Field | Usage | Notes |
|-------|-------|-------|
| `before_code` | **Input** | The code to review (primary input) |
| `reviewer_comment` | **Target Output** | What the model should generate |
| `diff_context` | **Optional Input** | Provides diff context (can improve understanding) |
| `language` | **Filter** | Filter for `language == "Python"` |
| `quality_score` | **Filter** | Filter for `quality_score >= 0.5` to ensure quality |
| `comment_type` | **Optional** | Can exclude "none" type (auto-approvals with no real comments) |

### Fields to Ignore for Training

| Field | Reason |
|-------|--------|
| `after_code` | Not needed for review generation (used for edit tasks) |
| `file_path` | Not relevant for review quality |
| `comment_line` | Metadata, not needed for training |
| `comment_length` | Derived from `reviewer_comment` |
| `before_lines` / `after_lines` | Metadata |
| `is_negative` | Not needed unless doing sentiment classification |
| `pr_title` | Metadata |
| `pr_number` | Metadata |
| `repo_name` | Metadata |
| `repo_stars` | Metadata |
| `repo_language` | Redundant with `language` |
| `reviewer_username` | Metadata (consider excluding Copilot reviews) |
| `author_username` | Metadata |

### Python-Only Dataset Stats

From the analysis, Python accounts for ~21.6% of the dataset:
- **Estimated Python rows in train**: ~72,000 rows
- **After quality filter (>= 0.5)**: ~50,000-60,000 rows
- **After excluding "none" comment_type**: ~40,000-50,000 rows

### Suggested Data Filtering Pipeline

```python
import polars as pl

df = pl.scan_parquet("dataset/github_codereview_train.parquet")

python_reviews = (
    df
    .filter(pl.col("language") == "Python")
    .filter(pl.col("quality_score") >= 0.5)
    .filter(pl.col("comment_type") != "none")  # Exclude auto-approvals
    .filter(pl.col("reviewer_username") != "Copilot")  # Optional: exclude AI reviews
    .select(["before_code", "reviewer_comment", "diff_context"])
    .collect()
)
```

### Sample Training Format

**Input prompt**:
```
Review the following Python code and provide constructive feedback:

```python
{before_code}
```

{diff_context}
```

**Target output**:
```
{reviewer_comment}
```

### Comment Type Distribution for Python

For filtering by comment type, the distribution for Python reviews is similar to overall:
- `suggestion`: ~47% - Code fix suggestions with markdown code blocks
- `question`: ~13% - Clarifying questions
- `bug`: ~7% - Bug identification
- `refactor`: ~4% - Refactoring suggestions
- `performance`: ~2% - Performance improvements
- `security`: ~2% - Security concerns
- `style`: ~1% - Code style issues
- `nitpick`: ~2% - Minor issues

### Quality Score Recommendations

| Score Range | Recommendation |
|-------------|----------------|
| 0.0 - 0.3 | Exclude (low quality) |
| 0.3 - 0.5 | Use cautiously |
| 0.5 - 0.7 | Good quality |
| 0.7 - 1.0 | Best quality |

For a small model, prefer `quality_score >= 0.6` to ensure high-quality training data.

### Training Data Size Estimates

| Filter | Estimated Rows |
|--------|----------------|
| Python only | ~72,000 |
| + quality >= 0.5 | ~50,000 |
| + exclude "none" type | ~40,000 |
| + exclude Copilot | ~36,000 |

This is a good size for fine-tuning a small Qwen model (e.g., Qwen-1.5B or Qwen-7B).
