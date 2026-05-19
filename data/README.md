# Data

This directory is reserved for dataset instructions and lightweight metadata only.

The Oxford Flowers-102 dataset should **not** be committed to the public repository. Keep the raw dataset locally and follow the preparation logic in the training scripts.

Suggested local structure:

```text
flowers102/
├── jpg/
├── imagelabels.mat
├── setid.mat
├── train/
├── val/
└── test/
```

For a publication-grade release, this folder may later include:
- dataset cards
- preprocessing notes
- sample index files
- split statistics
