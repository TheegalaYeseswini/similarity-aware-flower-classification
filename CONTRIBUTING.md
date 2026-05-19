# Contributing

Thank you for your interest in contributing to this repository.

## Scope
This project is maintained as a research-oriented implementation for fine-grained flower classification on Oxford Flowers-102. Contributions are most helpful when they improve one of the following:

- reproducibility
- code modularity
- training stability
- visualization quality
- documentation clarity
- evaluation rigor

## Local Setup
```bash
git clone https://github.com/TheegalaYeseswini/similarity-aware-flower-classification.git
cd similarity-aware-flower-classification
pip install -r requirements.txt
```

Large local assets such as datasets, checkpoints, and logs are intentionally excluded from version control.

## Recommended Contribution Areas
- refactoring shared training utilities into reusable modules under `src/`
- adding reproducible configuration files
- implementing additional baselines such as standalone focal loss
- extending visual diagnostics such as Grad-CAM or t-SNE plots
- improving evaluation scripts and experiment tracking

## Style Guidelines
- Prefer clear, descriptive names over short abbreviations.
- Preserve backwards compatibility for published training commands when possible.
- Add concise docstrings for new public functions and classes.
- Avoid committing datasets, checkpoints, or generated logs.
- Keep documentation research-oriented and technically precise.

## Pull Request Guidance
Please include:

1. a short problem statement
2. a summary of the proposed change
3. any experiment or validation evidence
4. notes on limitations or follow-up work

## Reporting Issues
When opening an issue, include:
- expected behavior
- observed behavior
- environment details
- minimal reproduction steps if applicable
