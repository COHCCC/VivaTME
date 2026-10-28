# Downloading the HTAN breast Visium sections

The HTAN breast cohort of the paper has 22 Visium sections from the Human Tumor Atlas Network (HTAN, WUSTL atlas `HTA12`). The files are hosted on Synapse and listed on the HTAN data portal (https://humantumoratlas.org/explore). Level-3/4 Visium data are open access, but you need a free Synapse account.

## 1. Set up the Synapse client

```bash
pip install synapseclient
synapse login            # use a personal access token created at synapse.org (Account Settings > Personal Access Tokens)
```

Never write the token into a script. `synapse login` stores it in `~/.synapseConfig`.

## 2. Download

`docs/download_htan_breast.sh` downloads 20 of the 22 sections by their Synapse IDs, 7 files per section: the H&E image(s), `barcodes.tsv.gz`, `features.tsv.gz`, `matrix.mtx.gz`, `filtered_feature_bc_matrix.h5`, the `spatial/` files and a metadata table.

```bash
mkdir -p htan_breast && bash docs/download_htan_breast.sh htan_breast
```

The remaining two sections, `HTA12_255_7` and `HTA12_262_2`, are not in that list. Find them on the HTAN data portal: filter Atlas = HTAN WUSTL, Organ = Breast, Assay = 10x Visium, then search the biospecimen IDs `HT565B1-S1H2Fp2U1` (HTA12_255_7) and `HT339B2-H1Fp1U1` (HTA12_262_2). Use the portal's "Download" button to get the `synapse get` commands, and put each section in its own folder named by its HTAN ID.

## 3. Expected layout

Each section folder needs the Space Ranger matrix, the spot positions and the full-resolution H&E image that the positions refer to:

```
htan_breast/HTA12_140_5/
  barcodes.tsv.gz  features.tsv.gz  matrix.mtx.gz        (or filtered_feature_bc_matrix/...)
  spatial/tissue_positions_list.csv                      (or tissue_positions.csv)
  C1_HT268B1-Th1H3Fc2U22.tif                             full-resolution H&E
```

Some sections ship two TIFFs. Use the one listed for that section in `docs/htan_breast_samples.tsv`, which is the image used in the paper. Copy that sheet next to the section folders and pass it to `encoder/prepare_visium.py --samples`. Relative paths in the sheet are resolved from the sheet's own folder.

## 4. Run

```bash
python encoder/prepare_visium.py --samples htan_breast/htan_breast_samples.tsv --out prepared/htan_breast --stage all
sbatch encoder/train_slurm.sh encoder/configs/pooled_discotme.yaml runs/htan_breast_encoder inputs/htan_breast   # edit paths in the config first
python scripts/run_vivatme.py run --config configs/htan_breast.yaml
```
