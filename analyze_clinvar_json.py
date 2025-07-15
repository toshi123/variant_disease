import json
from collections import defaultdict
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Hiragino Sans' # macOS向けの日本語フォントを指定
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import os
import argparse
import math

# 3文字アミノ酸コードと1文字アミノ酸コードの対応表
AMINO_ACID_MAP = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D",
    "Cys": "C", "Gln": "Q", "Glu": "E", "Gly": "G",
    "His": "H", "Ile": "I", "Leu": "L", "Lys": "K",
    "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S",
    "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
    "Ter": "*", "Stp": "*" 
}

# ヒートマップの軸で使用するアミノ酸の順序
AMINO_ACIDS_ORDERED = ['I', 'V', 'L', 'F', 'C', 'M', 'A', 'G', 'T', 'S', 'W', 'Y', 'P', 'H', 'E', 'Q', 'D', 'N', 'K', 'R']

def plot_mutation_heatmap(mutation_counts, amino_acids_ordered, title, pdf_filename, normalize=False, p_all=None):
    """
    アミノ酸変異の割合または規格化スコアをヒートマップで表示する。
    総計が5未満の変異は表示しない (NaNとして扱う)。
    """
    num_amino_acids = len(amino_acids_ordered)
    heatmap_data = np.full((num_amino_acids, num_amino_acids), np.nan)
    
    aa_to_index = {aa: i for i, aa in enumerate(amino_acids_ordered)}

    for mutation_key, counts in mutation_counts.items():
        parts = mutation_key.split('->')
        if len(parts) != 2: continue
        orig_aa, mut_aa = parts

        if orig_aa in aa_to_index and mut_aa in aa_to_index:
            idx_orig = aa_to_index[orig_aa]
            idx_mut = aa_to_index[mut_aa]
            
            pathogenic_count = counts.get("Disease", 0)
            benign_count = counts.get("Polymorphism", 0)
            total_classified = pathogenic_count + benign_count

            if total_classified >= 5:
                if normalize and p_all is not None and p_all > 0:
                    # 規格化スコア S_d = log(P_filtered / P_all) を計算
                    p_filtered = pathogenic_count / total_classified if total_classified > 0 else 0
                    if p_filtered > 0:
                        score = math.log(p_filtered / p_all)
                        heatmap_data[idx_orig, idx_mut] = score
                else:
                    # 通常の疾患割合を計算
                    if total_classified > 0:
                        pathogenic_ratio = pathogenic_count / total_classified
                        heatmap_data[idx_orig, idx_mut] = pathogenic_ratio
            
    colors = [(0, "green"), (0.5, "white"), (1, "magenta")]
    cmap_name = "custom_green_magenta"
    custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=256)
    custom_cmap.set_bad(color='lightgray')

    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 規格化の有無でカラースケールとラベルを動的に変更
    if normalize:
        max_abs_val = np.nanmax(np.abs(heatmap_data))
        if max_abs_val == 0 or not np.isfinite(max_abs_val): max_abs_val = 1.0
        cax = ax.imshow(heatmap_data, cmap=custom_cmap, vmin=-max_abs_val, vmax=max_abs_val, aspect='auto')
        cbar = fig.colorbar(cax)
        cbar.set_label('Normalized Score: log(P_filtered / P_all)')
    else:
        cax = ax.imshow(heatmap_data, cmap=custom_cmap, vmin=0, vmax=1, aspect='auto')
        cbar = fig.colorbar(cax, ticks=[0, 0.25, 0.5, 0.75, 1])
        cbar.ax.set_yticklabels(['0.0 (Polymorphism, Green)', '0.25', '0.5 (Mid, White)', '0.75', '1.0 (Disease, Magenta)'])
        cbar.set_label('Disease Ratio (Disease / (Disease + Polymorphism))')

    ax.set_xticks(np.arange(num_amino_acids))
    ax.set_yticks(np.arange(num_amino_acids))
    ax.set_xticklabels(amino_acids_ordered)
    ax.set_yticklabels(amino_acids_ordered)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    ax.set_xlabel("Mutated Amino Acid (変異後)")
    ax.set_ylabel("Original Amino Acid (変異前)")
    ax.set_title(title)
    
    ax.set_xticks(np.arange(num_amino_acids + 1) - .5, minor=True)
    ax.set_yticks(np.arange(num_amino_acids + 1) - .5, minor=True)
    ax.grid(which="minor", color="grey", linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", size=0)

    fig.tight_layout()
    plt.savefig(pdf_filename, format="pdf", bbox_inches="tight")
    print(f"\nヒートマップを '{pdf_filename}' に保存しました。")

def count_mutations_from_data(data, allowed_acs=None):
    """指定されたデータとACリストから変異をカウントするヘルパー関数"""
    mutation_counts = defaultdict(lambda: {"Disease": 0, "Polymorphism": 0})
    is_filtered = allowed_acs is not None

    for ac, variants in data.items():
        if is_filtered and ac not in allowed_acs:
            continue
        
        for variant_info in variants:
            clinical_sig = variant_info.get("Clinical_significance")
            if clinical_sig not in ["Disease", "Polymorphism"]:
                continue

            variant = variant_info.get("Variant", {})
            orig_aa_3 = variant.get("before")
            mut_aa_3 = variant.get("after")

            orig_aa_1 = AMINO_ACID_MAP.get(orig_aa_3)
            mut_aa_1 = AMINO_ACID_MAP.get(mut_aa_3)

            if not (orig_aa_1 and mut_aa_1):
                continue

            mutation_key = f"{orig_aa_1}->{mut_aa_1}"
            mutation_counts[mutation_key][clinical_sig] += 1
    return mutation_counts

def analyze_clinvar_json(json_filepath, ac_list_filepath=None, normalize=False):
    """
    ClinVarのJSONデータを解析し、変異の統計情報を計算してヒートマップを描画する。
    """
    # JSONファイルを先に読み込む
    try:
        with open(json_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"エラー: JSONファイルが見つかりません: {json_filepath}")
        return
    except json.JSONDecodeError:
        print(f"エラー: JSONファイルの形式が正しくありません: {json_filepath}")
        return

    # --- 1. データベース全体の統計を計算 (P_all) ---
    all_mutation_counts = count_mutations_from_data(data)
    total_disease_all = sum(counts["Disease"] for counts in all_mutation_counts.values())
    total_polymorphism_all = sum(counts["Polymorphism"] for counts in all_mutation_counts.values())
    total_classified_all = total_disease_all + total_polymorphism_all
    p_all = total_classified_all > 0 and total_disease_all / total_classified_all

    if normalize and not ac_list_filepath:
        print("警告: 規格化(-n)はACリスト(-a)によるフィルタリング時にのみ意味があります。通常の割合で処理を続行します。")
        normalize = False # フィルタリングなしでの規格化は無意味なためフラグを降ろす

    # --- 2. 対象となる変異データを決定 ---
    mutation_counts_for_report = all_mutation_counts
    is_filtered_by_ac = False
    
    if ac_list_filepath:
        is_filtered_by_ac = True
        try:
            with open(ac_list_filepath, 'r') as f_ac:
                allowed_acs = {line.strip() for line in f_ac if line.strip()}
            if not allowed_acs:
                print(f"警告: ACリストファイル {ac_list_filepath} が空です。全てのACを対象とします。")
                is_filtered_by_ac = False
            else:
                 mutation_counts_for_report = count_mutations_from_data(data, allowed_acs)
        except FileNotFoundError:
            print(f"エラー: ACリストファイルが見つかりません: {ac_list_filepath}")
            return
    
    # --- 3. レポートの表示 ---
    mutation_ratios = []
    for mutation, counts in mutation_counts_for_report.items():
        disease_count = counts["Disease"]
        polymorphism_count = counts["Polymorphism"]
        total_classified = disease_count + polymorphism_count

        if total_classified > 0:
            ratio = disease_count / total_classified
            mutation_ratios.append({
                "mutation": mutation, "disease_count": disease_count,
                "polymorphism_count": polymorphism_count, "total_classified": total_classified,
                "disease_ratio": ratio
            })

    sorted_ratios = sorted(mutation_ratios, key=lambda x: x["disease_ratio"], reverse=True)

    print(f"アミノ酸変異の疾患関連性分析 (ファイル: {os.path.basename(json_filepath)})")
    if is_filtered_by_ac:
        print(f"対象ACリスト: {os.path.basename(ac_list_filepath)}")
    print("=" * 60)
    print(f"{'変異':<10} {'Disease数':<10} {'Polymorphism数':<15} {'総計(分類済)':<15} {'疾患割合':<10}")
    print("-" * 60)
    for item in sorted_ratios:
        if item['total_classified'] >= 5:
            print(f"{item['mutation']:<10} {item['disease_count']:<10} {item['polymorphism_count']:<15} {item['total_classified']:<15} {item['disease_ratio']:.2%}")

    # --- 4. ヒートマップの作成 ---
    json_basename = os.path.splitext(os.path.basename(json_filepath))[0]
    title_suffix = "(All ACs, Total Count >= 5)"
    filename_suffix = "all_ACs"

    if is_filtered_by_ac:
        ac_list_basename = os.path.splitext(os.path.basename(ac_list_filepath))[0]
        title_suffix = f"(ACs from {os.path.basename(ac_list_filepath)}, Total Count >= 5)"
        filename_suffix = ac_list_basename

    if normalize and is_filtered_by_ac:
        title = f"Heatmap of Normalized Score {title_suffix}"
        pdf_filename = f"clinvar_heatmap_{json_basename}_{filename_suffix}_normalized.pdf"
    else:
        title = f"Heatmap of Disease Ratio {title_suffix}"
        pdf_filename = f"clinvar_heatmap_{json_basename}_{filename_suffix}.pdf"

    plot_mutation_heatmap(mutation_counts_for_report, AMINO_ACIDS_ORDERED, title, pdf_filename, normalize=(normalize and is_filtered_by_ac), p_all=p_all)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a heatmap of amino acid mutations from a ClinVar JSON file.")
    parser.add_argument("json_file", help="Path to the input ClinVar JSON file.")
    parser.add_argument("--ac_list_file", "-a", default=None, help="Optional: Path to a newline-delimited accession list file to filter the analysis.")
    parser.add_argument("--normalize", "-n", action='store_true', help="Use normalized score for the heatmap when filtering with an AC list.")
    
    args = parser.parse_args()
    
    analyze_clinvar_json(json_filepath=args.json_file, ac_list_filepath=args.ac_list_file, normalize=args.normalize) 