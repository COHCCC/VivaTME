"""Barcode-aligned cohort loading of prepared Visium inputs (layout written by prepare_visium.py).

Sparse counts stay sparse until a target and its neighbors are requested. Every
feature strategy uses the FULL candidate-gene total as its normalization
 denominator. No sample labels, raw images, or existing per-slide HVG arrays
are used to construct gene features.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from scipy.spatial import cKDTree
from torch.utils.data import Dataset, WeightedRandomSampler


def _hash_ids(values: Sequence[str]) -> str:
    return hashlib.sha256(('\n'.join(values) + '\n').encode()).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_ids(path: Path) -> list[str]:
    if path.suffix.lower() in {'.tsv', '.csv'}:
        frame = pd.read_csv(path, sep='\t' if path.suffix.lower() == '.tsv' else ',', dtype=str, keep_default_na=False)
        if 'gene_id' not in frame:
            raise ValueError(f'Gene table has no gene_id column: {path}')
        if 'input_column' in frame:
            columns = pd.to_numeric(frame.input_column, errors='raise').to_numpy()
            if not np.array_equal(columns, np.arange(len(frame))):
                raise ValueError(f'Gene input_column must match file order: {path}')
        return frame.gene_id.tolist()
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _unique(values, label):
    if not values or len(values) != len(set(values)) or any(not str(v) for v in values):
        raise ValueError(f'{label} must be nonempty and unique')


class CohortDataset(Dataset):
    """Read audited sparse gene counts and original-color patch paths.

    ``gene_list`` accepts a TXT/TSV path, a list-strategy stem under
    ``genes/lists``, or an explicit sequence of stable gene IDs. ``sample_ids``
    selects samples from the prepared manifest without refitting the gene set.
    Coordinates are the image-axis assay-grid-inferred micrometer coordinates,
    not verified scanner calibration. This class does not decode patch images.
    """

    def __init__(self, prepared_root, gene_list, sample_ids=None, max_neighbors=15,
                 radius_um=200.0, target_sum=10000):
        self.prepared_root = Path(prepared_root).expanduser().resolve()
        if int(max_neighbors) != max_neighbors or max_neighbors < 0:
            raise ValueError('max_neighbors must be a nonnegative integer')
        if not np.isfinite(radius_um) or radius_um < 0:
            raise ValueError('radius_um must be finite and nonnegative')
        if not np.isfinite(target_sum) or target_sum <= 0:
            raise ValueError('target_sum must be finite and positive')
        self.max_neighbors = int(max_neighbors)
        self.radius_um = float(radius_um)
        self.target_sum = float(target_sum)
        genes_root = self.prepared_root / 'genes'
        prepare_path = genes_root / 'cache/prepare.json'
        prepare = json.loads(prepare_path.read_text())
        available = list(prepare['sample_ids'])
        _unique(available, 'Prepared sample IDs')
        self.sample_ids = available if sample_ids is None else list(sample_ids)
        _unique(self.sample_ids, 'Selected sample IDs')
        unknown = sorted(set(self.sample_ids) - set(available))
        if unknown:
            raise ValueError(f'Samples absent from prepared cohort: {unknown}')
        axis_path = genes_root / 'cache/common_assayed_nonmitochondrial.txt'
        cache_genes = _read_ids(axis_path)
        _unique(cache_genes, 'Raw cache gene axis')
        self._cache_gene_ids = cache_genes
        axis_lookup = {g: i for i, g in enumerate(cache_genes)}
        full_path = genes_root / 'lists/full_common_nonmitochondrial_nonzero.txt'
        if not full_path.exists():
            full_path = full_path.with_suffix('.tsv')
        full_genes = _read_ids(full_path)
        _unique(full_genes, 'Full candidate gene axis')
        if not set(full_genes).issubset(axis_lookup):
            raise ValueError('Full candidate list contains genes absent from raw cache axis')
        self._denominator_columns = np.asarray([axis_lookup[g] for g in full_genes], dtype=np.int64)
        list_path = None
        if isinstance(gene_list, (str, Path)):
            given = Path(gene_list).expanduser()
            options = [given, self.prepared_root / given, genes_root / 'lists' / given,
                       genes_root / 'lists' / (str(given) + '.txt'), genes_root / 'lists' / (str(given) + '.tsv')]
            list_path = next((p.resolve() for p in options if p.is_file()), None)
            if list_path is None:
                raise FileNotFoundError(f'Gene list not found: {gene_list}')
            self.gene_ids = _read_ids(list_path)
        else:
            self.gene_ids = list(gene_list)
        _unique(self.gene_ids, 'Model input genes')
        if not set(self.gene_ids).issubset(set(full_genes)):
            bad = sorted(set(self.gene_ids) - set(full_genes))
            raise ValueError(f'Model genes outside full candidate axis: {bad[:10]}')
        self.gene_dim = len(self.gene_ids)
        self._selected_columns = np.asarray([axis_lookup[g] for g in self.gene_ids], dtype=np.int64)
        self._samples = []
        self.sample_sizes = []
        self.spot_ids = []
        sample_records = []
        for sample_index, sid in enumerate(self.sample_ids):
            cache_dir = genes_root / 'cache' / sid
            counts_path = cache_dir / 'counts_common_nonmt.npz'
            rows_path = cache_dir / 'retained_spots.tsv'
            images_path = self.prepared_root / 'images' / sid / 'valid_spots.tsv'
            counts = sparse.load_npz(counts_path).tocsr()
            if counts.ndim != 2 or counts.shape[1] != len(cache_genes):
                raise ValueError(f'Raw matrix gene dimension mismatch: {sid}')
            if not np.isfinite(counts.data).all() or (counts.data < 0).any():
                raise ValueError(f'Nonfinite/negative raw counts: {sid}')
            if not np.equal(counts.data, np.floor(counts.data)).all():
                raise ValueError(f'Cache must contain raw integer counts: {sid}')
            rows = pd.read_csv(rows_path, sep='\t', dtype=str, keep_default_na=False)
            images = pd.read_csv(images_path, sep='\t', dtype=str, keep_default_na=False)
            if 'barcode' not in rows or len(rows) != counts.shape[0]:
                raise ValueError(f'Counts and retained-barcode rows differ: {sid}')
            _unique(rows.barcode.tolist(), f'{sid} retained barcodes')
            required = {'barcode', 'patch_path', 'x_um_grid_inferred', 'y_um_grid_inferred'}
            if not required.issubset(images.columns):
                raise ValueError(f'Missing image metadata fields for {sid}: {sorted(required-set(images.columns))}')
            _unique(images.barcode.tolist(), f'{sid} image barcodes')
            if 'sample_id' in images and not images.sample_id.eq(sid).all():
                raise ValueError(f'Image table contains another sample: {sid}')
            expected_ids = [f'{sid}::{barcode}' for barcode in rows.barcode]
            if 'spot_id' in rows and rows.spot_id.tolist() != expected_ids:
                raise ValueError(f'Noncanonical retained spot IDs: {sid}')
            if 'spot_id' in images and images.spot_id.tolist() != [f'{sid}::{b}' for b in images.barcode]:
                raise ValueError(f'Noncanonical image spot IDs: {sid}')
            # Never zip image-table rows to matrix rows: current production
            # tables have different order in all ten samples.
            matrix_index = pd.DataFrame({'barcode': rows.barcode, 'matrix_row': np.arange(len(rows))})
            joined = matrix_index.merge(images, on='barcode', how='left', sort=False, validate='one_to_one', indicator=True)
            if not joined['_merge'].eq('both').all():
                raise ValueError(f'Retained expression barcodes have no valid images: {sid}')
            joined = joined.sort_values('matrix_row', kind='stable').reset_index(drop=True)
            coordinates = joined[['x_um_grid_inferred', 'y_um_grid_inferred']].astype(np.float64).to_numpy()
            if not np.isfinite(coordinates).all():
                raise ValueError(f'Nonfinite physical coordinates: {sid}')
            paths = []
            for value in joined.patch_path:
                if not value:
                    raise ValueError(f'Empty patch path: {sid}')
                p = Path(value)
                p = p if p.is_absolute() else self.prepared_root / p
                if not p.is_file():
                    raise FileNotFoundError(p)
                paths.append(str(p.resolve()))
            totals = np.asarray(counts[:, self._denominator_columns].sum(axis=1)).ravel().astype(np.float64)
            if not np.isfinite(totals).all() or (totals <= 0).any():
                raise ValueError(f'Nonpositive full-candidate count denominator: {sid}')
            tree = cKDTree(coordinates)
            neighbors = []
            for row_index, xy in enumerate(coordinates):
                candidates = tree.query_ball_point(xy, r=self.radius_um)
                candidates = [j for j in candidates if j != row_index]
                ordered = sorted(candidates, key=lambda j: (float(np.linalg.norm(coordinates[j]-xy)), expected_ids[j]))
                neighbors.append(np.asarray(ordered[:self.max_neighbors], dtype=np.int64))
            self._samples.append({'counts': counts, 'totals': totals, 'coordinates': coordinates,
                                  'paths': paths, 'neighbors': neighbors, 'spot_ids': expected_ids})
            self.sample_sizes.append(len(rows))
            self.spot_ids.extend(expected_ids)
            sample_records.append({'sample_id': sid, 'n_spots': len(rows), 'counts_path': str(counts_path),
                                   'retained_spots_sha256': _file_hash(rows_path), 'valid_spots_sha256': _file_hash(images_path)})
        self._offsets = np.concatenate(([0], np.cumsum(self.sample_sizes))).astype(np.int64)
        self.sample_offsets = tuple(int(v) for v in self._offsets)
        self.sample_ranges = {sid: (self.sample_offsets[i], self.sample_offsets[i+1])
                              for i, sid in enumerate(self.sample_ids)}
        self.sample_indices = np.repeat(np.arange(len(self.sample_ids), dtype=np.int64), self.sample_sizes)
        self.input_contract = {
            'schema_version': 'cohort_dataset_v1', 'prepared_root': str(self.prepared_root),
            'gene_dim': self.gene_dim, 'gene_ids_sha256': _hash_ids(self.gene_ids),
            'gene_list_path': None if list_path is None else str(list_path),
            'gene_list_file_sha256': None if list_path is None else _file_hash(list_path),
            'raw_cache_gene_axis_sha256': _hash_ids(cache_genes),
            'full_denominator_gene_ids_sha256': _hash_ids(full_genes),
            'full_denominator_n_genes': len(full_genes), 'prepare_sha256': _file_hash(prepare_path),
            'sample_ids': list(self.sample_ids), 'sample_records': sample_records,
            'normalization': {'target_sum': self.target_sum, 'transform': 'log1p',
                              'denominator': 'full_common_nonmitochondrial_nonzero raw counts before selected-gene subsetting'},
            'filter': 'Prepared retained gene spots joined one-to-one to successful in-tissue image patches; no additional low-UMI/appearance thresholds',
            'neighbors': {'max_neighbors': self.max_neighbors, 'radius_um': self.radius_um,
                          'within_sample_only': True, 'self_excluded': True, 'padding': 'zeros plus false mask; empty path',
                          'tie_break': 'distance then canonical spot_id'},
            'coordinates': 'Image-axis x_um_grid_inferred,y_um_grid_inferred; assay-grid inference, not scanner calibration',
            'feature_dtype': 'float32', 'mask_dtype': 'bool', 'sample_index_dtype_collated': 'int64',
            'sample_selection_does_not_refit_gene_lists': True,
        }

    def __len__(self):
        return int(self._offsets[-1])

    def __getitem__(self, index):
        index = int(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        sample_index = int(np.searchsorted(self._offsets, index, side='right')-1)
        row = index-int(self._offsets[sample_index])
        sample = self._samples[sample_index]
        neighbors = sample['neighbors'][row]
        selected_rows = np.concatenate(([row], neighbors))
        # Selection commutes with a fixed row scale and log1p. The denominator
        # was computed on FULL candidates above, never the selected gene list.
        values = sample['counts'][selected_rows][:, self._selected_columns].toarray().astype(np.float32)
        scales = (self.target_sum/sample['totals'][selected_rows]).astype(np.float32)
        values *= scales[:, None]
        np.log1p(values, out=values)
        count = len(neighbors)
        context = np.zeros((self.max_neighbors, self.gene_dim), dtype=np.float32)
        relative = np.zeros((self.max_neighbors, 2), dtype=np.float32)
        mask = np.zeros(self.max_neighbors, dtype=bool)
        paths = ['']*self.max_neighbors
        if count:
            context[:count] = values[1:]
            relative[:count] = sample['coordinates'][neighbors]-sample['coordinates'][row]
            mask[:count] = True
            paths[:count] = [sample['paths'][j] for j in neighbors]
        return {'target_gene': torch.from_numpy(values[0]), 'context_genes': torch.from_numpy(context),
                'relative_pos_um': torch.from_numpy(relative), 'neighbor_mask': torch.from_numpy(mask),
                'sample_index': sample_index, 'spot_id': sample['spot_ids'][row],
                'target_patch_path': sample['paths'][row], 'context_patch_paths': paths}


def collate_cohort(items):
    """Stack tensors; preserve patch paths as [B] and [B][K] Python lists."""
    if not items:
        raise ValueError('Cannot collate an empty batch')
    output = {key: torch.stack([item[key] for item in items]) for key in
              ['target_gene', 'context_genes', 'relative_pos_um', 'neighbor_mask']}
    output['sample_index'] = torch.tensor([item['sample_index'] for item in items], dtype=torch.long)
    for key in ['spot_id', 'target_patch_path', 'context_patch_paths']:
        output[key] = [item[key] for item in items]
    return output


def make_balanced_sampler(dataset: CohortDataset, num_samples, seed):
    """Equal total sampling mass per sample; repeats are explicitly allowed."""
    if int(num_samples) != num_samples or num_samples < 1:
        raise ValueError('num_samples must be a positive integer')
    counts = np.asarray(dataset.sample_sizes, dtype=np.float64)
    weights = 1.0/counts[dataset.sample_indices]
    generator = torch.Generator().manual_seed(int(seed))
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double),
                                 num_samples=int(num_samples), replacement=True, generator=generator)


from PIL import Image


class RGBTrainingDataset(Dataset):
    """Decode only real RGB patches and attach stable cohort-wide spot indexes."""
    def __init__(self, cohort):
        self.cohort = cohort
        self.mean = torch.tensor([.485, .456, .406])[:, None, None]
        self.std = torch.tensor([.229, .224, .225])[:, None, None]

    def __len__(self):
        return len(self.cohort)

    def read_patch(self, path):
        with Image.open(path) as image:
            if image.size != (224, 224):
                raise ValueError(f'Expected preprocessed 224x224 patch: {path}, {image.size}')
            array = np.asarray(image.convert('RGB'), dtype=np.uint8).copy()
        tensor = torch.from_numpy(array).permute(2, 0, 1).float().div_(255.)
        return (tensor-self.mean)/self.std

    def __getitem__(self, index):
        item = self.cohort[index]
        item['target_img'] = self.read_patch(item['target_patch_path'])
        context = torch.zeros(len(item['context_patch_paths']), 3, 224, 224)
        for j, path in enumerate(item['context_patch_paths']):
            if bool(item['neighbor_mask'][j]):
                if not path:
                    raise ValueError('Valid context has empty image path')
                context[j] = self.read_patch(path)
        item['context_imgs'] = context
        item['global_spot_index'] = int(index)
        return item


def collate_training(items):
    batch = collate_cohort(items)
    for key in ['target_img', 'context_imgs']:
        batch[key] = torch.stack([item[key] for item in items])
    batch['global_spot_index'] = torch.tensor([item['global_spot_index'] for item in items], dtype=torch.long)
    return batch
