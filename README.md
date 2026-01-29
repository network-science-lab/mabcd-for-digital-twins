# mABCD for digital twins of complex systems

A repository for...

**Authors**:  (†),  (¶),  (¶),  (¬), (†),  (§)

- (¶)
- (†)
- (¬)
- (§)

This repository is a complementary artifact for the [paper]().

## Structure of the Repository

```bash
.
├── README.md
├── src                      -> Main code used by various scripts
└── ...  -> ...
```

## Runtime Configuration

1. Install [uv](https://uv.run/) as the project is built with it.
2. Install the project dependencies:
   ```bash
   uv sync
   ```
3. Additionally, to use DVC with Google Drive as remote storage, install:
   ```bash
   uv tool install dvc[gdrive]
   ```
4. Download the data required for experiments using DVC:
   ```bash
   dvc pull
   ```

## Usage

To run the code execute: `uv run mfdt <path to the configuration file>`. To debug in VS Code, see
`.vscode/launch.json` for an example configuration as `uv` is not widely supported yet by IDEs.

There're three main functionalities provided by this repository:
1. **mABCD Generator**: Generates mABCD twins for given networks.
2. **mABCD Finder**: Finds mABCD twins for given networks.
3. **Proximity Evaluator**: Evaluates the proximity between original networks and their mABCD twins.

### 1. mABCD Generator

See `scripts/configs/example_generate_1.yaml` for an example configuration file where all
parameters are defined explicitly. See `scripts/configs/example_generate_2` for an example where
parameters are read from another file, e.g., have been estimated with the finder.

### 2. mABCD Finder

See `scripts/configs/example_find.yaml` for an example configuration file that estimates mABCD
parameters for the provided networks.

### 3. Proximity Evaluator

See `scripts/configs/example_evaluate.yaml` for an example configuration file that assess quality 
of the mABCD parameters found by the finder.

## Experiments

Finder experiments ():

- exp_a: rudimentary
- exp_b: fancy with [r] and d=2
- exp_c: fancy with [r, tau] and d=2
- exp_d: fancy with [r] and d=1
- exp_e: fancy with [r] and d=4
- exp_f: fancy with [r] and d=8
- exp_g: fancy with [tau] and d=2

Twins generation:

- experiment_d: rudimentary d = {1,2,4,8} and fancy d = {1,2,4,8}
- experiment_finder_method: rudimentary, fancy with [t, tau], fancy with [r], fancy with [tau], d=2

Evaluation experiments:

- experiment_d: rudimentary d = {1,2,4,8} and fancy d = {1,2,4,8}
- experiment_finder_method: rudimentary, fancy with [t, tau], fancy with [r], fancy with [tau], d=2

## Acknowledgment

This work was supported by the...


TODO: 
- align loss with paper
- enrich logging to log diver. tau and r ALWAYS
- add better loss choice
- reorganise code
- add D to the estimator!