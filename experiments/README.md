# Experiments

This directory is intended for structured experiment manifests such as:

- YAML or JSON configuration files
- trial summaries
- ablation manifests
- experiment registry tables

The current repository still contains legacy root-level training scripts. A future refactor should move shared training logic into `src/` and preserve this directory for declarative experiment management.
