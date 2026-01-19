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

First, ...

## Usage

To run the code execute: `python run_experiments.py <path to the configuration file>`.

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

TO DO!!!

## Acknowledgment

This work was supported by the...


## Doodles

- add uv
- migrate to the server

improve estimating r
how to handle batches of networks?
where can we assess quality of estimation?
- during the estimation process and generate a report along with the parameters
- after the estimation process using the proximity evaluator
