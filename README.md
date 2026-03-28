# `mABCD` for digital twins of complex networked systems

A repository for retrieving digital twins of a given network using `mABCD`, a synthetic graph
generator: from an empirical network to an `mABCD` configuration and its corresponding digital twin.

**Authors**: Piotr Bródka (¶†),  Michał Czuba(¶†),  Łukasz Kraiński (¬),  Bogumił Kamiński (¬),
Katarzyna Musial (†), Paweł Prałat (§), Mateusz Stolarski (¶)

**Affiliations**:
- (¶) Dept. of Atrificial Intelligence, Wrocław University of Science and Technology, Wrocław, PL
- (†) Data Science Institute, University of Technology Sydney, Sydney, AU
- (¬) Decision Analysis and Support Unit, SGH Warsaw School of Economics, Warsaw, PL
- (§) Dept. of Mathematics, Toronto Metropolitan University, Toronto, CA

This repository is a complementary artifact for the [paper](https://arxiv.org/abs/2602.02044)
presented at the 21st Workshop on Modelling and Mining Networks
([WAW 2026](https://math.torontomu.ca/waw2026/)), Toronto, ON, Canada, 15-19 June 2026.

## Structure of the Repository

```bash
.
├── README.md
├── data                           ->  DVC directory
│   ├── evaluate                   ->  evaluation results
│   ├── finder                     ->  retrieved twins for Freebase
│   └── networks                   ->  raw network data
├── pyproject.toml                 ->  project's configuratioon
├── scripts                        ->  auxiliary files, not the project's logic
│   ├── analysis                   ->  scripts to transform obtained results
│   └── configs                    ->  exemplary configs to run the project
└── src
    └── mfdt
        ├── config_finder          ->  module with logic to find twins
        ├── correlations           ->  aux. scripts with various correlations
        ├── divergences.py         ->  divergency scores
        ├── evaluator.py           ->  job to calculate twin's divergency
        ├── finder.py              ->  job to find twins of a given network
        ├── generator.py           ->  job to create mABCD networks
        ├── loaders                ->  network loaders
        ├── main.py                ->  main entrypoint to the project
        ├── mln_abcd               ->  Python ports to Julia impl. mABCD
        ├── params_handler.py
        └── utils.py
```

## Runtime Configuration

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/). `uv` manages the Python
   environment and dependency installation for this project.
2. From the repository root, sync the project environment and install dependencies:
   ```bash
   uv sync
   ```
3. Resolve Julia dependencies (required on first setup):
   ```bash
   uv run python -c "import juliapkg; juliapkg.resolve(force=True)"
   ```
4. Additionally, to use DVC with Google Drive as remote storage, install:
   ```bash
   uv tool install 'dvc[gdrive]'
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

Experiment 1:
- exp_a: optimise [r]; loss [r]; d = 1
- exp_b: optimise [r]; loss [r]; d = 2
- exp_c: optimise [r]; loss [r]; d = 4
- exp_d: optimise [r]; loss [r]; d = 8

Experiment 2:
- exp_e: optimise [r, tau]; loss [r]; d = 2
- exp_f: optimise [r, tau]; loss [tau]; d = 2
- exp_g: optimise [r, tau]; loss [r+tau]; d = 2

Experiment 3:
- exp_h: exp_b
- exp_i: optimise [r, d]; loss [r]

## Acknowledgment

This research was partially supported by: (1) EU under the Horizon Europe, grant no. 101086321
OMINO; (2) Polish Ministry of Science and Higher Education, International Projects Co-Funded
programme; (3) National Science Centre, Poland, grant no. 2022/45/B/ST6/04145; (4) Polish National
Agency for Academic Exchange, Strategic Partnerships programme, grant no.
BPI/PST/2024/1/00129/U/00001; (5) Wrocław University of Science and Technology, Academia Profesorum
Iuniorum programme. Views and opinions expressed are, however, those of the authors only and do not
necessarily reflect those of the founding agencies. 
