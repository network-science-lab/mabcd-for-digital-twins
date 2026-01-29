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

Experiment 1:
- exp_a: optimise [r]; loss [r]; d = 1
- exp_b: optimise [r]; loss [r]; d = 2
- exp_c: optimise [r]; loss [r]; d = 4
- exp_d: optimise [r]; loss [r]; d = 8

Experiment 2:
- exp_e: optimise [r, tau]; loss [r]
- exp_f: optimise [r, tau]; loss [tau]
- exp_g: optimise [r, tau]; loss [r+tau]

Experiment 3:
- exp_h: exp_b
- exp_i: optimise [r, d]; loss [r]

## Acknowledgment

This work was supported by the...
