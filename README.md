# VivaTME

**VivaTME** (multi-**V**iew **I**ntegration and **V**ariation **A**nalysis of **T**umor **M**icro**E**nvironments) aligns tumor-microenvironment compartments across tissue sections and quantifies, for every compartment in every section, a calibrated deviation from its shared state.

Given a cohort of spatial transcriptomics sections with matched H&E images, VivaTME:

0. encodes every spot with a multimodal, multiscale model trained jointly on all sections (pooled [DiSCoTME](https://github.com/COHCCC/DiSCoTME), `encoder/`);
1. fits **shared compartments with whole-section offsets** on these frozen spot representations;
2. aligns the representations **within compartments** with a residual adapter trained against compartment-conditional section discriminators (gradient reversal). Only confidently assigned spots with a cross-section mutual nearest neighbour take part in this alignment;
3. reads out compartment labels from the aligned representation, then decomposes each compartment mean of each section on the **pre-alignment** representation into a shared center, a section-wide shift and a sparse deviation;
4. sets the reporting threshold τ by a **full-refit permutation null**: the whole pipeline is rerun on 500 block-permuted cohorts, which bounds the family-wise error rate at 0.05.

This repository contains the code that produces the VivaTME compartments and deviations reported in the paper, starting from Visium Space Ranger output and H&E images.

## Installation

```bash
conda create -n vivatme python=3.9.18
conda activate vivatme
pip install -r requirements.txt
```

VivaTME itself runs on CPU (Linux) with the versions pinned in `requirements.txt`; only the encoder (`requirements-encoder.txt`) needs a CUDA GPU.

## From Visium sections to U (`encoder/`)

VivaTME works on frozen multimodal spot representations `U` from a DiSCoTME model trained jointly on all sections of a cohort ("pooled DiSCoTME"). The model architecture is DiSCoTME's (`encoder/dilated_blocks.py` is copied unchanged from that repository). `encoder/` holds the three steps from raw Visium data to `U`:

```bash
pip install -r requirements-encoder.txt          # needs a CUDA GPU (LongNet attention via torchscale/xformers)

# 1. Preprocessing (CPU). Sample sheet: sample_id, spaceranger_dir, image (full-resolution H&E TIFF)
python encoder/prepare_visium.py --samples samples.tsv --out prepared/mycohort --stage all

# 2. Joint training on all sections + 3. export of U  (the paper used 4 nodes x 4 V100, 10 epochs)
sbatch encoder/train_slurm.sh encoder/configs/pooled_discotme.yaml runs/mycohort_encoder inputs/mycohort
```

- **Preprocessing.** Spot coordinates are converted to micrometres by fitting the Visium lattice (100 µm centre pitch). Each in-tissue spot gets a 55 × 55 µm H&E crop, resized to 224 × 224 px in its original colours. Genes are the Ensembl IDs assayed in every section, with mitochondrial genes and genes without counts in the cohort removed. Counts are normalised to 10,000 per spot and log-transformed when they are loaded.
- **Training.** An image branch (DINO ViT-S/16, trainable) and a gene branch (residual MLP) are each followed by a dilated-attention (LongNet) context encoder over the 15 nearest spots within 200 µm, plus an adaptive local/neighbourhood gate. The two branches are trained with a symmetric multi-positive InfoNCE loss with count-matched mixed negatives: 10 epochs, seed 42, global batch 256, AdamW, cosine schedule, AMP. Section-balanced sampling gives every section equal weight.
- **Export.** `U = [F_img/‖F_img‖, F_gene/‖F_gene‖]`, where F is the 256-d output of each context encoder before the projection heads.

`docs/download_htan_breast.md` describes how to download the HTAN breast sections used in the paper.

## Input of VivaTME

VivaTME runs on two files per cohort; `encoder/extract.py` writes both:

| file | content |
| --- | --- |
| `U.npy` | `n_spots × 512` float array: `[f_img/‖f_img‖, f_gene/‖f_gene‖]`, the L2-normalised image and gene branch outputs (256 each) of a DiSCoTME model trained jointly on all sections of the cohort and then frozen |
| `spots.csv` | one row per spot, in the row order of `U.npy`, with the columns `global_index` (0…n−1), `spot_id` (unique), `sample_id`, `sample_index` (0…S−1, rows sorted by it) and `x_um`, `y_um` (spot centre in µm). Only these columns are read, so annotations never enter the model. |

Cohorts used in the paper:

| config | cohort | sections | K | raw data |
| --- | --- | --- | --- | --- |
| `configs/her2.yaml` | HER2-positive breast cancer (Andersson et al. 2021) | 36 | 6 | Zenodo [10.5281/zenodo.4751624](https://doi.org/10.5281/zenodo.4751624) |
| `configs/crc.yaml` | Colorectal cancer (Valdeolivas et al. 2024) | 14 | 8 | Zenodo [7760264](https://zenodo.org/records/7760264) |
| `configs/tnbc.yaml` | Triple-negative breast cancer (Wang et al. 2024) | 94 | 9 | Zenodo [14204217](https://zenodo.org/records/14204217) |
| `configs/htan_breast.yaml` | HTAN breast, Visium | 22 | 6 | HTAN data portal |
| `configs/htan_crc.yaml` | HTAN colorectal, Visium | 12 | 6 | HTAN data portal |

`encoder/prepare_visium.py` handles standard Visium (both HTAN cohorts). HER2 and TNBC are legacy Spatial Transcriptomics arrays, and the CRC sections came with NDPI scans. For these three cohorts, patches and coordinates were prepared with the same rules (crop of one spot field of view, resized to 224 px; micrometre coordinates from the array pitch) by dataset-specific readers that are not part of this repository.

Point the configs at your data with two environment variables:

```bash
export VIVATME_DATA=/path/to/inputs      # contains <cohort>/U.npy and <cohort>/spots.csv
export VIVATME_OUT=/path/to/results
```

## Running

All stages run on CPU. The first three commands below reproduce one cohort.

```bash
C=configs/her2.yaml

# 1. Reference fit on U: shared compartments (10 k-means++ restarts), spatial blocks, neighbour graphs
python scripts/run_vivatme.py prepare --config $C

# 2. Adapter training and compartment read-out; seeds 43 and 44 share the seed-42 reference
for s in 42 43 44; do python scripts/run_vivatme.py train --config $C --seed $s; done

# 3. Full-refit null: 500 independent permutations (one SLURM array task each), then tau
python scripts/run_vivatme.py null --config $C --index 0      # ... --index 499
python scripts/run_vivatme.py summarize --config $C

# 4. Deviations of every seed at the calibrated tau95
for s in 42 43 44; do python scripts/run_vivatme.py deviations --config $C --seed $s; done
```

`scripts/slurm_example.sh` chains these stages on a SLURM cluster. `--smoke` runs a short schedule (15 epochs, 2 permutations) to test the code path.

Approximate cost per cohort: `prepare` takes minutes, and each `train` seed takes 0.5–3 h on 8 CPU cores. Each null permutation reruns the whole pipeline and takes about 6–60 min on 8 cores with < 11 GB of memory; the median was 6 min (HER2) and 59 min (TNBC).

### Outputs (`$VIVATME_OUT/<cohort>/`)

| path | content |
| --- | --- |
| `reference/fit.npz` | reference fit on U: `labels` (−1 = dustbin), `centers_full`, `offsets`, `q` |
| `reference/graphs.npz` | within-section 15-NN, cross-section MNN participation flag, spatial block of every spot |
| `seed{S}/labels.csv` | compartment of every spot (`label`: native id, −1 = dustbin; `compartment`: reference compartment `C0…`) |
| `seed{S}/compartments.npz` | read-out fit on the aligned representation |
| `seed{S}/train/` | EMA adapter weights (`model_ema.pt`), aligned features, training log |
| `null/perm_XXX.json`, `null/thresholds.json` | maximum z of each permutation; τ90/τ95/τ99 |
| `seed{S}/deviations.npz` | `delta` (K × S × 512), `z` (K × S), `eligible` (soft mass ≥ 30), `residual` r_sk, `section_mass`, `centers_full`, `offsets`, `tau`. A cell (k, s) is reported when it is eligible and z > τ (equivalently ‖δ‖ > 0). |

Full-refit thresholds obtained in the paper (τ95, 500 permutations): HER2 2.356, CRC 5.025, TNBC 3.463, HTAN breast 8.333, HTAN CRC 8.285.

## Settings

All hyper-parameters are fixed in the code and listed in the Methods:

- **Compartments:** soft assignment with dustbin cost c0 (95th percentile of the initial nearest-centre distance) and temperature T (0.1 × its median); section-balanced centres; offsets from compartments present (soft mass ≥ 30) in ≥ 3 sections, λ_o = 10; tolerance 1e-4.
- **Adapter:** 512–256–512 SELU, zero-initialised output; L_keep with 15 near and 15 far within-section spots; 0.1‖g‖²; 0.1 L_proto. One discriminator per compartment (512–128–S, dropout 0.2, label smoothing 0.1), active with ≥ 3 sections × ≥ 20 confident (q ≥ 0.6) MNN spots; 32 spots per (compartment, section) and step. Batch 1,024, 100 epochs, AdamW (1e-3 / 3e-3, weight decay 0.01), adversary from epoch 10 with λ(t) rising to 10; EMA 0.999.
- **Deviations:** group soft-threshold with √m_sk scaling, eligible cells m_sk ≥ 30. The fixed-assignment fit runs twice, first from the reference and then from its own result.
- **Calibration:** blocks of ~25 spots (≤ 550 µm) within each (section, compartment); 500 permutations; τ = 95th percentile of the maximum z.

## Reproducibility

The package is a consolidation of the research code used for the paper. On the same inputs it reproduces the original reference fit, spatial blocks and neighbour graphs exactly, and floating-point values to within 1e-14. Training with PyTorch on CPU is deterministic for a given machine and thread count, but can differ slightly across hardware and library builds. Compartment labels and deviations are therefore reproduced up to this numerical noise; cross-seed agreement is reported in the paper.

## Citation

> VivaTME: Multimodal Alignment for Quantifying Patient Differences within Shared Tumor Microenvironment Compartments. (manuscript in preparation, 2026)

Spot representations: Song J. *et al.* Using multi-scale spatial integration of histology and transcriptomics to discover hidden tumor microenvironment motifs. https://github.com/COHCCC/DiSCoTME

## License

MIT (see `LICENSE`).
