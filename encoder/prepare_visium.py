"""Prepare a cohort of Visium sections (Space Ranger output + full-resolution H&E) for pooled DiSCoTME.

Input: a tab-separated sample sheet with the columns
  sample_id        unique section name ([A-Za-z0-9_.-]+)
  spaceranger_dir  Space Ranger 'outs' folder (or a flat folder) holding filtered_feature_bc_matrix/{barcodes,features,matrix}
                   (.tsv/.mtx, optionally .gz) and spatial/tissue_positions[_list].csv
  image            full-resolution H&E TIFF (uint8 RGB) the positions refer to
Output (OUT): images/<sid>/{patches_rgb_224/*.png, valid_spots.tsv, geometry.json} and
              genes/{cache/prepare.json, cache/common_assayed_nonmitochondrial.txt, cache/<sid>/counts_common_nonmt.npz,
              cache/<sid>/retained_spots.tsv, lists/full_common_nonmitochondrial_nonzero.{tsv,txt}}

Stages (each idempotent; 'images' and 'genes-sample' can run per section in parallel):
  images        per section: fit the Visium lattice (100 um centre pitch) to the spot pixels, giving micrometre coordinates and
                the pixel scale; crop a 55 x 55 um square around every in-tissue spot (bilinear, fractional bounds) -> 224 x 224
                PNG in original colours. Spots whose crop leaves the image are excluded.
  genes-axis    cohort: gene axis = Ensembl gene IDs assayed in every section, mitochondrial (MT-) genes removed.
  genes-sample  per section: raw integer counts on that axis for spots that are in tissue, have a patch and nonzero counts.
  genes-finalize cohort: drop genes with zero counts in the whole cohort -> the model's gene list.
usage: python encoder/prepare_visium.py --samples samples.tsv --out OUT --stage all  [--sample-id SID]
"""
from __future__ import annotations
import argparse
import gzip
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from PIL import Image
from scipy import io, sparse

Image.MAX_IMAGE_PIXELS = None
POSITION_COLUMNS = ['barcode', 'in_tissue', 'array_row', 'array_col', 'pxl_row_in_fullres', 'pxl_col_in_fullres']
PITCH_UM, PATCH_UM, PATCH_PX = 100., 55., 224


def write_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp'); tmp.write_text(json.dumps(obj, indent=2) + '\n'); tmp.replace(path)


def write_table(frame, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep='\t', index=False, float_format='%.17g')


def read_samples(path):
    frame = pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False)
    for c in ('sample_id', 'spaceranger_dir', 'image'):
        if c not in frame:
            raise ValueError(f'sample sheet needs column {c}')
    if frame.sample_id.duplicated().any() or not frame.sample_id.str.fullmatch(r'[A-Za-z0-9_.-]+').all():
        raise ValueError('sample_id must be unique and match [A-Za-z0-9_.-]+')
    base = Path(path).resolve().parent
    for c in ('spaceranger_dir', 'image'):
        frame[c] = [str(Path(v) if Path(v).is_absolute() else (base / v).resolve()) for v in frame[c]]
    return frame.to_dict('records')


def positions_path(raw):
    found = [p for p in (Path(raw) / 'spatial' / n for n in ('tissue_positions.csv', 'tissue_positions_list.csv')) if p.is_file()]
    if len(found) != 1:
        raise ValueError(f'Ambiguous or absent tissue positions in {raw}/spatial')
    return found[0]


def locate(raw, name):
    options = [Path(raw) / name, Path(raw) / (name + '.gz'), Path(raw) / 'filtered_feature_bc_matrix' / name,
               Path(raw) / 'filtered_feature_bc_matrix' / (name + '.gz')]
    present = [p for p in options if p.is_file()]
    if len(present) != 1:
        raise ValueError(f'Expected one {name} in {raw}; found {present}')
    return present[0]


# ----------------------------------------------------------------------------- images
def read_positions_px(path):
    frame = pd.read_csv(path, header=None, dtype=str, keep_default_na=False)
    if str(frame.iloc[0, 0]).lstrip('﻿').lower() == 'barcode':
        frame.columns = [c.strip().lstrip('﻿') for c in frame.iloc[0]]; frame = frame.iloc[1:]
    else:
        frame.columns = POSITION_COLUMNS
    rows = []
    for r in frame[POSITION_COLUMNS].itertuples(index=False):
        rows.append(dict(barcode=r[0], in_tissue=int(r[1]), array_row=int(r[2]), array_col=int(r[3]), x=float(r[5]), y=float(r[4])))
    if len({r['barcode'] for r in rows}) != len(rows):
        raise ValueError('Duplicate barcode in positions table')
    return rows


def fit_grid(rows, pitch):
    """Affine map from the standard Visium hexagonal lattice (micrometres) to full-resolution pixels; robust refit."""
    u = np.array([[r['array_col'] * pitch / 2, r['array_row'] * pitch * math.sqrt(3) / 2, 1.] for r in rows])
    xy = np.array([[r['x'], r['y']] for r in rows])
    keep = np.ones(len(rows), dtype=bool)
    for _ in range(5):
        b = np.linalg.lstsq(u[keep], xy[keep], rcond=None)[0]
        residual = np.linalg.norm(xy - u @ b, axis=1)
        med = np.median(residual[keep]); mad = np.median(abs(residual[keep] - med))
        new_keep = residual <= max(1.5, med + 6 * 1.4826 * mad)
        if np.array_equal(keep, new_keep):
            break
        if new_keep.sum() < max(100, len(rows) * .9):
            raise ValueError('More than 10% of positions deviate from the standard Visium lattice')
        keep = new_keep
    b = np.linalg.lstsq(u[keep], xy[keep], rcond=None)[0]
    residual = np.linalg.norm(xy - u @ b, axis=1)
    singular = np.linalg.svd(b[:2], compute_uv=False)
    mpp = np.linalg.norm(np.linalg.inv(b[:2]), axis=1)
    rmse = float(np.sqrt(np.mean(residual[keep] ** 2)))
    if rmse > 2.0 or singular.max() / singular.min() > 1.1:
        raise ValueError(f'Grid fit needs manual review: RMSE {rmse:.3f} px, anisotropy {singular.max() / singular.min():.3f}')
    return dict(mpp_x=float(mpp[0]), mpp_y=float(mpp[1]), rmse_px=rmse, affine_um_to_px=b.tolist())


def load_pixels(path):
    with tifffile.TiffFile(path) as tf:
        page = tf.pages[0]; series = tf.series[0]
        if page.dtype != np.dtype('uint8') or series.axes not in ('YXS', 'YXC') or series.shape[-1] != 3:
            raise ValueError('Expected a uint8 RGB level-0 TIFF')
        orientation = page.tags.get('Orientation')
        if orientation is not None and orientation.value not in (1,):
            raise ValueError('Non-default TIFF orientation requires explicit coordinate handling')
        memmappable = bool(page.is_memmappable); shape = series.shape
    if memmappable:
        return tifffile.memmap(path, series=0, level=0), shape
    with Image.open(path) as im:
        if im.mode != 'RGB':
            raise ValueError(f'Expected RGB TIFF, found {im.mode}')
        return np.array(im, dtype=np.uint8), shape


def patch_from_array(arr, x0, y0, x1, y1, size=PATCH_PX):
    left, top = math.floor(x0), math.floor(y0); right, bottom = math.ceil(x1), math.ceil(y1)
    im = Image.fromarray(np.array(arr[top:bottom, left:right], copy=True))
    return im.transform((size, size), Image.Transform.EXTENT, (x0 - left, y0 - top, x1 - left, y1 - top), resample=Image.Resampling.BILINEAR)


def stage_images(row, out):
    sid = row['sample_id']; dest = out / 'images' / sid
    if (dest / 'done.json').exists():
        return
    (dest / 'patches_rgb_224').mkdir(parents=True, exist_ok=True)
    rows = read_positions_px(positions_path(row['spaceranger_dir']))
    geom = fit_grid(rows, PITCH_UM)
    arr, shape = load_pixels(row['image']); h, w = shape[:2]
    mx, my = geom['mpp_x'], geom['mpp_y']; side_x, side_y = PATCH_UM / mx, PATCH_UM / my
    valid = []
    for r in rows:
        x, y = r['x'], r['y']; x0, x1, y0, y1 = x - side_x / 2, x + side_x / 2, y - side_y / 2, y + side_y / 2
        ok = (r['in_tissue'] == 1 and math.isfinite(x) and math.isfinite(y) and 0 <= x < w and 0 <= y < h
              and x0 >= 0 and y0 >= 0 and x1 <= w and y1 <= h)
        if not ok:
            continue
        p = dest / 'patches_rgb_224' / f"{r['barcode']}.png"
        patch_from_array(arr, x0, y0, x1, y1).save(p, format='PNG', compress_level=2)
        valid.append(dict(barcode=r['barcode'], spot_id=f"{sid}::{r['barcode']}", sample_id=sid, x_px=x, y_px=y,
                          x_um_grid_inferred=x * mx, y_um_grid_inferred=y * my, patch_path=str(p)))
    write_table(pd.DataFrame(valid), dest / 'valid_spots.tsv')
    write_json(dest / 'geometry.json', dict(geom, sample_id=sid, image=row['image'], image_shape=list(shape), patch_side_um=PATCH_UM,
                                            n_positions=len(rows), n_in_tissue=sum(r['in_tissue'] for r in rows), n_patches=len(valid)))
    write_json(dest / 'done.json', dict(status='complete', n_patches=len(valid)))
    print(sid, 'patches', len(valid), 'grid mpp', round(mx, 4), round(my, 4), flush=True)


# ----------------------------------------------------------------------------- genes
def read_features(raw):
    path = locate(raw, 'features.tsv')
    frame = pd.read_csv(path, sep='\t', header=None, dtype=str, keep_default_na=False)
    frame.columns = ['raw_gene_id', 'gene_symbol'] + (['feature_type'] if frame.shape[1] == 3 else [])
    if 'feature_type' not in frame:
        frame['feature_type'] = 'Gene Expression'
    frame['gene_id'] = frame.raw_gene_id.str.replace(r'^(ENSG[0-9]+)\.[0-9]+$', r'\1', regex=True)
    gene = frame.feature_type.eq('Gene Expression')
    recognized = frame.gene_id.str.fullmatch(r'ENSG[0-9]+') | frame.raw_gene_id.str.fullmatch(r'DEPRECATED_ENSG[0-9]+(?:\.[0-9]+)?')
    if not recognized.loc[gene].all():
        raise ValueError(f'Unrecognized human gene IDs in {path}')
    if frame.loc[gene, 'gene_id'].duplicated().any():
        raise ValueError(f'Duplicate stable gene IDs in {path}')
    return frame


def stage_genes_axis(samples, out):
    catalog = {}
    for row in samples:
        f = read_features(row['spaceranger_dir'])
        for rec in f[f.feature_type.eq('Gene Expression')].to_dict('records'):
            c = catalog.setdefault(rec['gene_id'], dict(samples=set(), symbols=set()))
            c['samples'].add(row['sample_id']); c['symbols'].add(rec['gene_symbol'])
    ids = [r['sample_id'] for r in samples]; genes, symbols = [], {}
    for gid, c in sorted(catalog.items()):
        if gid.startswith('DEPRECATED_') or len(c['samples']) < len(ids) or any(s.upper().startswith('MT-') for s in c['symbols']):
            continue
        genes.append(gid); symbols[gid] = ';'.join(sorted(c['symbols']))
    cache = out / 'genes/cache'; cache.mkdir(parents=True, exist_ok=True)
    (cache / 'common_assayed_nonmitochondrial.txt').write_text('\n'.join(genes) + '\n')
    pd.DataFrame(dict(gene_id=genes, gene_symbols=[symbols[g] for g in genes])).to_csv(cache / 'gene_symbols.tsv', sep='\t', index=False)
    write_json(cache / 'prepare.json', dict(sample_ids=ids, n_common_nonmitochondrial=len(genes)))
    print('gene axis', len(genes), 'genes common to', len(ids), 'sections', flush=True)


def stage_genes_sample(row, out):
    sid = row['sample_id']; d = out / 'genes/cache' / sid
    if (d / 'retained_spots.tsv').exists():
        return
    d.mkdir(parents=True, exist_ok=True)
    genes = (out / 'genes/cache/common_assayed_nonmitochondrial.txt').read_text().splitlines()
    features = read_features(row['spaceranger_dir'])
    barcodes = pd.read_csv(locate(row['spaceranger_dir'], 'barcodes.tsv'), sep='\t', header=None, dtype=str)[0]
    matrix_path = locate(row['spaceranger_dir'], 'matrix.mtx')
    with (gzip.open if matrix_path.suffix == '.gz' else open)(matrix_path, 'rb') as handle:
        counts = io.mmread(handle).tocsr().T.tocsr()
    counts.sum_duplicates(); counts.eliminate_zeros()
    if counts.shape != (len(barcodes), len(features)):
        raise ValueError(f'Matrix shape differs from barcodes/features: {sid}')
    gene_cols = np.flatnonzero(features.feature_type.eq('Gene Expression').to_numpy())
    lookup = {features.iloc[i].gene_id: i for i in gene_cols}
    candidate = counts[:, np.array([lookup[g] for g in genes])].astype(np.float64)
    total = np.asarray(candidate.sum(axis=1)).ravel()
    pos = read_positions_px(positions_path(row['spaceranger_dir'])); pos = {r['barcode']: r for r in pos}
    valid = set(pd.read_csv(out / 'images' / sid / 'valid_spots.tsv', sep='\t', dtype=str, keep_default_na=False).barcode)
    keep = np.array([b in pos and pos[b]['in_tissue'] == 1 and math.isfinite(pos[b]['x']) and math.isfinite(pos[b]['y'])
                     and b in valid and t > 0 for b, t in zip(barcodes, total)])
    if keep.sum() < 3:
        raise ValueError(f'Fewer than three valid tissue spots: {sid}')
    kept = candidate[keep].tocsr()
    dtype = np.int32 if kept.data.max(initial=0) <= np.iinfo(np.int32).max else np.int64
    sparse.save_npz(d / 'counts_common_nonmt.npz', kept.astype(dtype), compressed=False)
    np.save(d / 'gene_totals.npy', np.asarray(kept.sum(axis=0)).ravel())
    write_table(pd.DataFrame(dict(barcode=barcodes[keep].to_numpy(), spot_id=sid + '::' + barcodes[keep].to_numpy())), d / 'retained_spots.tsv')
    print(sid, 'retained spots', int(keep.sum()), flush=True)


def stage_genes_finalize(samples, out):
    cache = out / 'genes/cache'; genes = np.array((cache / 'common_assayed_nonmitochondrial.txt').read_text().splitlines())
    totals = sum(np.load(cache / r['sample_id'] / 'gene_totals.npy') for r in samples)
    keep = totals > 0
    sym = pd.read_csv(cache / 'gene_symbols.tsv', sep='\t', dtype=str, keep_default_na=False).set_index('gene_id').gene_symbols
    lst = out / 'genes/lists'; lst.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(dict(input_column=np.arange(int(keep.sum())), gene_id=genes[keep], gene_symbols=sym.loc[genes[keep]].to_numpy()))
    write_table(frame, lst / 'full_common_nonmitochondrial_nonzero.tsv')
    (lst / 'full_common_nonmitochondrial_nonzero.txt').write_text('\n'.join(frame.gene_id) + '\n')
    print('model gene list', len(frame), 'genes', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--samples', required=True, type=Path); ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--stage', choices=['images', 'genes-axis', 'genes-sample', 'genes-finalize', 'all'], default='all')
    ap.add_argument('--sample-id', help='run a per-section stage for one section only')
    a = ap.parse_args()
    samples = read_samples(a.samples); a.out.mkdir(parents=True, exist_ok=True)
    chosen = [r for r in samples if a.sample_id in (None, r['sample_id'])]
    if a.stage in ('images', 'all'):
        for r in chosen:
            stage_images(r, a.out)
    if a.stage in ('genes-axis', 'all'):
        stage_genes_axis(samples, a.out)
    if a.stage in ('genes-sample', 'all'):
        for r in chosen:
            stage_genes_sample(r, a.out)
    if a.stage in ('genes-finalize', 'all'):
        stage_genes_finalize(samples, a.out)


if __name__ == '__main__':
    main()
