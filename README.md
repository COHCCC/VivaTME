# VivaTME

**VivaTME** (multi-**V**iew **I**ntegration and **V**ariation **A**nalysis of **T**umor **M**icro**E**nvironments) takes a cohort of Visium sections from different patients and returns:

- **shared tissue compartments** across all sections and patients (one label per spot);
- for every compartment in every section, a **deviation** from the compartment's shared state, with a calibrated reporting threshold.

## Quick start

```bash
git clone https://github.com/COHCCC/VivaTME.git && cd VivaTME
pip install -r requirements-encoder.txt
```

**1. List your sections** in a tab-separated sheet. Each section needs its Space Ranger output folder and the full-resolution H&E image:

```
sample_id	spaceranger_dir	image
P1_A	data/P1_A/outs	data/P1_A/HE.tif
P2_A	data/P2_A/outs	data/P2_A/HE.tif
...
```

**2. Encode the spots** (multimodal, multiscale DiSCoTME trained jointly on all sections):

```bash
python encoder/prepare_visium.py --samples samples.tsv --out prepared/mycohort
sbatch encoder/train_slurm.sh encoder/configs/pooled_discotme.yaml runs/mycohort inputs/mycohort
```

Before submitting, set `prepared_root`, `gene_list` and the DINO weight path in the config. The job writes `inputs/mycohort/U.npy` and `spots.csv`.

**3. Run VivaTME:**

```bash
cp configs/htan_breast.yaml configs/mycohort.yaml    # set cohort: mycohort and K (number of compartments)
export VIVATME_DATA=inputs VIVATME_OUT=results
python scripts/run_vivatme.py run --config configs/mycohort.yaml
```

Results are in `results/mycohort/seed42/`:

| file | content |
| --- | --- |
| `labels.csv` | `spot_id, sample_id, compartment` (`C0` … `C{K-1}`, or `dustbin` for spots that fit no compartment) |
| `deviations.csv` | one row per section × compartment: `soft_mass`, standardized deviation `z`, `deviation_norm`, `reported`, sorted by `z` |

## Calibrated threshold

`run` reports deviations with a fast threshold from the initial fit. The threshold used in the paper comes from a **full-refit permutation null**: the whole pipeline is rerun on 500 permuted cohorts, which bounds the family-wise error rate over all section × compartment cells at 0.05. The permutations are independent, so run them as an array job:

```bash
sbatch --array=0-499 --wrap "python scripts/run_vivatme.py null --config configs/mycohort.yaml"
python scripts/run_vivatme.py summarize --config configs/mycohort.yaml      # writes null/thresholds.json
python scripts/run_vivatme.py run --config configs/mycohort.yaml            # deviations at the calibrated tau
```

`scripts/slurm_example.sh` chains all stages on a SLURM cluster.

## How it works

**Encoder.** Each spot's 224 × 224 px H&E patch and its gene expression go through an image branch (DINO ViT-S/16) and a gene branch. Each branch has a dilated-attention context encoder over the 15 nearest spots within 200 µm and an adaptive gate between local and neighbourhood information. The two branches are trained contrastively on all sections of the cohort. The spot representation is

```math
\mathbf{u}_i = \left[\, \mathbf{f}^{\mathrm{img}}_i / \lVert \mathbf{f}^{\mathrm{img}}_i \rVert ,\; \mathbf{f}^{\mathrm{gene}}_i / \lVert \mathbf{f}^{\mathrm{gene}}_i \rVert \,\right] \in \mathbb{R}^{512}.
```

**Shared compartments.** $K$ centres $\boldsymbol\mu_k$, a whole-section offset $\mathbf{o}_s$ and a dustbin are fitted with soft assignments

```math
q_{ik} \propto \exp\!\left(-\lVert \mathbf{u}_i - \mathbf{o}_{s(i)} - \boldsymbol\mu_k \rVert^2 / T\right).
```

**Alignment within compartments.** A residual adapter $\mathbf{z}_i = \mathbf{u}_i + g(\mathbf{u}_i)$ preserves local geometry. One discriminator per compartment tries to predict the section of $\mathbf{z}_i - \boldsymbol\eta_{s(i)}$, and the adapter is trained against it through gradient reversal. Only confidently assigned spots with a mutual nearest neighbour in another section take part, so a compartment can be shared by a subset of sections. Compartment labels are read out from the aligned representation.

**Deviations.** With these labels held fixed, each compartment mean on the pre-alignment representation is decomposed as

```math
\bar{\mathbf{u}}_{sk} \approx \boldsymbol\mu_k + \mathbf{o}_s + \boldsymbol\delta_{sk},
\qquad
\boldsymbol\delta_{sk} = \mathbf{r}_{sk}\left(1 - \frac{\tau}{z_{sk}}\right)_{+},
\qquad
z_{sk} = \sqrt{m_{sk}}\,\lVert \mathbf{r}_{sk} \rVert ,
```

where $\mathbf{r}_{sk}$ is the residual after the shared centre and the section offset and $m_{sk}$ is the soft mass of the cell. A deviation is reported when $z_{sk} > \tau$.

## Details

### Inputs

| file | content |
| --- | --- |
| sample sheet | `sample_id` ([A-Za-z0-9_.-]+), `spaceranger_dir` (holds `filtered_feature_bc_matrix/` and `spatial/tissue_positions[_list].csv`), `image` (full-resolution uint8 RGB TIFF). Relative paths are resolved from the sheet's folder. |
| `U.npy` | `n_spots × 512` spot representations (written by `encoder/extract.py`) |
| `spots.csv` | `global_index`, `spot_id`, `sample_id`, `sample_index`, `x_um`, `y_um`, in the row order of `U.npy`. Any `U.npy` + `spots.csv` pair in this format can be used directly in step 3. |
| `configs/*.yaml` | `cohort`, `K`, `U`, `spots` (relative to `$VIVATME_DATA`), `out` |

### Outputs (`results/<cohort>/`)

| path | content |
| --- | --- |
| `seed42/labels.csv`, `seed42/deviations.csv` | main results (see Quick start) |
| `seed42/deviations.npz` | arrays: `delta` (K × S × 512), `z`, `eligible` (soft mass ≥ 30), `residual`, `section_mass`, `centers_full`, `offsets`, `tau` |
| `seed42/compartments.npz`, `seed42/train/` | compartment fit on the aligned representation; adapter weights and aligned features |
| `reference/` | initial compartment fit, spatial blocks and neighbour graphs |
| `null/` | maximum z of every permutation and `thresholds.json` (τ90 / τ95 / τ99) |

### Preprocessing (`encoder/prepare_visium.py`)

| step | rule |
| --- | --- |
| coordinates | affine fit of the Visium hexagonal lattice (100 µm centre pitch) to the spot pixels, giving µm per pixel |
| patches | 55 × 55 µm crop around every in-tissue spot, bilinear resampling to 224 × 224 px, original colours; spots whose crop leaves the image are excluded |
| genes | Ensembl IDs assayed in every section, without mitochondrial genes and without genes that have zero counts in the whole cohort |
| expression | counts per spot scaled to 10,000, then `log1p` |

### Encoder (`encoder/configs/pooled_discotme.yaml`)

| setting | value |
| --- | --- |
| image branch | DINO ViT-S/16 (trainable) → 256-d |
| gene branch | residual MLP → 512-d |
| context encoder | LongNet dilated attention, 4 layers, 8 heads, 15 neighbours within 200 µm; adaptive local/neighbourhood gate |
| loss | symmetric multi-positive InfoNCE, temperature 0.07, count-matched mixed negatives |
| training | 10 epochs, global batch 256, section-balanced sampling, AdamW with cosine schedule, mixed precision |

### VivaTME (`vivatme/`)

| component | setting |
| --- | --- |
| compartments | dustbin cost: 95th percentile of the initial nearest-centre distance; $T$ = 0.1 × its median; offsets from compartments present (soft mass ≥ 30) in ≥ 3 sections, ridge λ = 10 |
| adapter | 512–256–512 SELU, zero-initialised output; geometry loss on 15 near and 15 far within-section spots; 0.1 ‖g‖²; 0.1 prototype loss |
| discriminators | one per compartment, 512–128–S, dropout 0.2, label smoothing 0.1; active with ≥ 3 sections × ≥ 20 confident (q ≥ 0.6) spots that have a mutual nearest neighbour; 32 spots per section and step |
| training | batch 1,024, 100 epochs, AdamW (1e-3 adapter, 3e-3 discriminators), adversary from epoch 10 with weight rising to λ = 10, EMA 0.999 |
| deviations | cells with soft mass ≥ 30; fixed-assignment group soft-threshold |
| calibration | blocks of ~25 spots (≤ 550 µm) within each section × compartment; 500 permutations; τ = 95th percentile of the maximum z |

### Cohorts of the paper

| config | cohort | sections | K | full-refit τ95 |
| --- | --- | --- | --- | --- |
| `her2.yaml` | HER2-positive breast cancer (Andersson et al. 2021, [Zenodo 4751624](https://doi.org/10.5281/zenodo.4751624)) | 36 | 6 | 2.356 |
| `crc.yaml` | colorectal cancer (Valdeolivas et al. 2024, [Zenodo 7760264](https://zenodo.org/records/7760264)) | 14 | 8 | 5.025 |
| `tnbc.yaml` | triple-negative breast cancer (Wang et al. 2024, [Zenodo 14204217](https://zenodo.org/records/14204217)) | 94 | 9 | 3.463 |
| `htan_breast.yaml` | HTAN breast, Visium ([download](docs/download_htan_breast.md)) | 22 | 6 | 8.333 |
| `htan_crc.yaml` | HTAN colorectal, Visium | 12 | 6 | 8.285 |

HER2 and TNBC are legacy Spatial Transcriptomics arrays and the CRC sections came with NDPI scans. Their patches and coordinates were prepared with the same rules by dataset-specific readers.

## Citation

> VivaTME: Multimodal Alignment for Quantifying Patient Differences within Shared Tumor Microenvironment Compartments. Manuscript in preparation (2026).

The encoder architecture is from DiSCoTME: https://github.com/COHCCC/DiSCoTME

## License

MIT (see `LICENSE`).
