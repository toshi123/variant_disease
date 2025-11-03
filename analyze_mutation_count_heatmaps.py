import json
import argparse
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams['font.family'] = 'Hiragino Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42

AMINO_ACID_MAP = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D",
    "Cys": "C", "Gln": "Q", "Glu": "E", "Gly": "G",
    "His": "H", "Ile": "I", "Leu": "L", "Lys": "K",
    "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S",
    "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
    "Ter": "*", "Stp": "*"
}

AMINO_ACIDS_ORDERED = ['I', 'V', 'L', 'F', 'C', 'M', 'A', 'G', 'T', 'S', 'W', 'Y', 'P', 'H', 'E', 'Q', 'D', 'N', 'K', 'R']


def save_heatmap_as_tsv(heatmap_data: np.ndarray, amino_acids_ordered: List[str], tsv_filename: str):
    """
    ヒートマップの数値データをTSV形式で保存する。
    """
    with open(tsv_filename, 'w', encoding='utf-8') as f:
        # ヘッダー行を書き込み
        f.write("Original_AA")
        for aa in amino_acids_ordered:
            f.write(f"\t{aa}")
        f.write("\n")
        
        # データ行を書き込み
        for i, orig_aa in enumerate(amino_acids_ordered):
            f.write(orig_aa)
            for j in range(len(amino_acids_ordered)):
                value = heatmap_data[i, j]
                if np.isnan(value):
                    f.write("\tNA")
                else:
                    f.write(f"\t{value:.6f}")
            f.write("\n")
    
    print(f"ヒートマップデータを '{tsv_filename}' に保存しました。")


def load_json(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_ac_list(path: str) -> List[str]:
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def count_mutations(data: Dict[str, list], allowed_acs: Optional[set] = None) -> Dict[str, int]:
    """
    Count mutations (A->B) across the dataset or a filtered subset (by ACs).
    Counts ALL variants regardless of Clinical_significance.
    Returns dict: key 'A->B' -> count
    """
    counts: Dict[str, int] = defaultdict(int)
    is_filtered = allowed_acs is not None

    for ac, variants in data.items():
        if is_filtered and ac not in allowed_acs:
            continue
        for v in variants:
            var = v.get('Variant', {})
            before_3 = var.get('before')
            after_3 = var.get('after')
            before = AMINO_ACID_MAP.get(before_3)
            after = AMINO_ACID_MAP.get(after_3)
            if not before or not after:
                continue
            if before == '*' or after == '*':  # skip stop
                continue
            key = f"{before}->{after}"
            counts[key] += 1
    return counts


def counts_to_matrix(counts: Dict[str, int], min_count: int = 5) -> np.ndarray:
    n = len(AMINO_ACIDS_ORDERED)
    mat = np.full((n, n), np.nan, dtype=float)  # NaNで初期化（グレー表示用）
    idx = {aa: i for i, aa in enumerate(AMINO_ACIDS_ORDERED)}
    for key, c in counts.items():
        try:
            a, b = key.split('->')
        except ValueError:
            continue
        if a in idx and b in idx:
            if c >= min_count:  # 最小カウント以上の場合のみ値を設定
                mat[idx[a], idx[b]] = c
    return mat


def plot_count_heatmap(counts_subset: Dict[str, int], title: str, pdf_path: str, max_count: Optional[float] = None):
    n = len(AMINO_ACIDS_ORDERED)
    data = counts_to_matrix(counts_subset, min_count=5)
    vmax = max_count if max_count is not None else (np.nanmax(data) if data.size else 1.0)
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    # TSVファイルとして数値データを保存
    tsv_path = pdf_path.replace('.pdf', '.tsv')
    save_heatmap_as_tsv(data, AMINO_ACIDS_ORDERED, tsv_path)

    # White -> Blue colormap
    colors = [(1, 1, 1), (0.1, 0.2, 0.8)]
    cmap = LinearSegmentedColormap.from_list('white_blue', colors, N=256)
    cmap.set_bad(color='lightgray')  # NaN値をグレーで表示

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=vmax, aspect='auto')

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(AMINO_ACIDS_ORDERED)
    ax.set_yticklabels(AMINO_ACIDS_ORDERED)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    ax.set_xlabel('Mutated AA (after)')
    ax.set_ylabel('Original AA (before)')
    ax.set_title(title)

    cbar = fig.colorbar(im)
    cbar.set_label('Count')

    ax.set_xticks(np.arange(n + 1) - .5, minor=True)
    ax.set_yticks(np.arange(n + 1) - .5, minor=True)
    ax.grid(which='minor', color='grey', linestyle='-', linewidth=0.5)
    ax.tick_params(which='minor', size=0)

    fig.tight_layout()
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"保存: {pdf_path}")


def plot_normalized_log_ratio(counts_all: Dict[str, int], counts_subset: Dict[str, int], title: str, pdf_path: str, max_abs: Optional[float] = None):
    # Compute shares
    total_all = sum(counts_all.values())
    total_sub = sum(counts_subset.values())
    n = len(AMINO_ACIDS_ORDERED)
    data = np.full((n, n), np.nan)
    idx = {aa: i for i, aa in enumerate(AMINO_ACIDS_ORDERED)}

    for a in AMINO_ACIDS_ORDERED:
        for b in AMINO_ACIDS_ORDERED:
            key = f"{a}->{b}"
            ca = counts_all.get(key, 0)
            cs = counts_subset.get(key, 0)
            if total_all == 0 or total_sub == 0:
                continue
            # 5つ未満の変異は除外（グレー表示）
            if ca < 5 or cs < 5:
                continue
            p_all = ca / total_all
            p_sub = cs / total_sub
            if p_all <= 0 or p_sub <= 0:
                continue
            score = np.log(p_sub / p_all)
            data[idx[a], idx[b]] = score

    # TSVファイルとして数値データを保存
    tsv_path = pdf_path.replace('.pdf', '.tsv')
    save_heatmap_as_tsv(data, AMINO_ACIDS_ORDERED, tsv_path)

    # Diverging Blue -> White -> Orange
    colors = [(0.1, 0.2, 0.8), (1, 1, 1), (1.0, 0.55, 0.0)]
    cmap = LinearSegmentedColormap.from_list('blue_white_orange', colors, N=256)
    cmap.set_bad(color='lightgray')  # NaN値をグレーで表示

    if max_abs is None:
        max_abs = np.nanmax(np.abs(data))
        if not np.isfinite(max_abs) or max_abs <= 0:
            max_abs = 1.0

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(data, cmap=cmap, vmin=-max_abs, vmax=max_abs, aspect='auto')

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(AMINO_ACIDS_ORDERED)
    ax.set_yticklabels(AMINO_ACIDS_ORDERED)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    ax.set_xlabel('Mutated AA (after)')
    ax.set_ylabel('Original AA (before)')
    ax.set_title(title)

    cbar = fig.colorbar(im)
    cbar.set_label('log( share_subset / share_all )')

    ax.set_xticks(np.arange(n + 1) - .5, minor=True)
    ax.set_yticks(np.arange(n + 1) - .5, minor=True)
    ax.grid(which='minor', color='grey', linestyle='-', linewidth=0.5)
    ax.tick_params(which='minor', size=0)

    fig.tight_layout()
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"保存: {pdf_path}")


def main():
    parser = argparse.ArgumentParser(description='Create two heatmaps: (1) raw mutation counts in subset; (2) normalized log ratio vs full dataset.')
    parser.add_argument('clinvar_json', help='Path to ClinVar JSON (AC -> list of variants)')
    parser.add_argument('-a', '--ac_list_file', required=True, help='Newline-delimited AC list to include for subset')
    parser.add_argument('-c', '--max_count', type=float, default=None, help='Max color scale for count heatmap (vmax).')
    parser.add_argument('-m', '--max_abs_value', type=float, default=None, help='Absolute max for normalized heatmap symmetric scale.')

    args = parser.parse_args()

    data = load_json(args.clinvar_json)
    allowed_acs = set(load_ac_list(args.ac_list_file))

    counts_all = count_mutations(data, allowed_acs=None)
    counts_sub = count_mutations(data, allowed_acs=allowed_acs)

    base = os.path.splitext(os.path.basename(args.clinvar_json))[0]
    acl = os.path.splitext(os.path.basename(args.ac_list_file))[0]

    # Count heatmap
    count_pdf = f"HeatmapCounts_{base}_{acl}.pdf" if args.max_count is None else f"HeatmapCounts_{base}_{acl}_{args.max_count}.pdf"
    plot_count_heatmap(counts_sub, f"Mutation counts (subset: {acl})", count_pdf, max_count=args.max_count)

    # Normalized heatmap
    norm_pdf = f"HeatmapNormalized_{base}_{acl}.pdf" if args.max_abs_value is None else f"HeatmapNormalized_{base}_{acl}_{args.max_abs_value}.pdf"
    plot_normalized_log_ratio(counts_all, counts_sub, f"Normalized log-ratio vs all (subset: {acl})", norm_pdf, max_abs=args.max_abs_value)


if __name__ == '__main__':
    main()
