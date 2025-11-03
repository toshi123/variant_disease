#!/usr/bin/env python3
"""
膜貫通領域に存在する変異と領域外に存在する変異のヒートマップを生成するスクリプト
膜貫通回数ごとに別々のヒートマップを作成（1回膜貫通型から15回膜貫通型まで）
"""

import json
from collections import defaultdict
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Hiragino Sans' # macOS向けの日本語フォントを指定
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import os
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

def is_position_in_transmembrane_regions(position, transmembrane_regions):
    """
    指定された位置が膜貫通領域内にあるかどうかを判定する
    
    Args:
        position: アミノ酸位置（整数または文字列）
        transmembrane_regions: 膜貫通領域のリスト [{"start": int, "end": int}, ...]
    
    Returns:
        bool: 位置が膜貫通領域内にある場合True
    """
    if not isinstance(transmembrane_regions, list):
        return False
    
    try:
        pos = int(position)
    except (ValueError, TypeError):
        return False
    
    for region in transmembrane_regions:
        if not isinstance(region, dict):
            continue
        try:
            start = int(region.get("start", 0))
            end = int(region.get("end", 0))
            if start <= pos <= end:
                return True
        except (ValueError, TypeError):
            continue
    return False

def save_heatmap_as_tsv(heatmap_data, amino_acids_ordered, tsv_filename):
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

def plot_mutation_heatmap(mutation_counts, amino_acids_ordered, title, pdf_filename, normalize=False, p_all=None, max_abs_value=None):
    """
    アミノ酸変異の割合または規格化スコアをヒートマップで表示する。
    総計が5未満の変異は表示しない (NaNとして扱う)。
    """
    num_amino_acids = len(amino_acids_ordered)
    heatmap_data = np.full((num_amino_acids, num_amino_acids), np.nan)
    
    aa_to_index = {aa: i for i, aa in enumerate(amino_acids_ordered)}

    for mutation_key, counts in mutation_counts.items():
        if not isinstance(counts, dict):
            continue
        
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
    
    # TSVファイルとして数値データを保存
    tsv_filename = pdf_filename.replace('.pdf', '.tsv')
    save_heatmap_as_tsv(heatmap_data, amino_acids_ordered, tsv_filename)
            
    colors = [(0, "green"), (0.5, "white"), (1, "magenta")]
    cmap_name = "custom_green_magenta"
    custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=256)
    custom_cmap.set_bad(color='lightgray')

    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 規格化の有無でカラースケールとラベルを動的に変更
    if normalize:
        if max_abs_value is not None:
            # ユーザー指定の最大値を使用
            max_val = max_abs_value
        else:
            # データから自動計算（従来の動作）
            max_val = np.nanmax(np.abs(heatmap_data))
            if max_val == 0 or not np.isfinite(max_val): 
                max_val = 1.0
        
        cax = ax.imshow(heatmap_data, cmap=custom_cmap, vmin=-max_val, vmax=max_val, aspect='auto')
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
    plt.close()  # メモリリークを防ぐため閉じる
    print(f"\nヒートマップを '{pdf_filename}' に保存しました。")

def count_mutations_from_data(data, transmembrane_data, allowed_acs=None, in_transmembrane=True):
    """
    指定されたデータとACリストから変異をカウントするヘルパー関数
    膜貫通領域内または領域外の変異のみをフィルタリングする
    
    Args:
        data: ClinVar JSONデータ
        transmembrane_data: 膜貫通領域データ
        allowed_acs: 許可するACのセット（Noneの場合は全て）
        in_transmembrane: Trueの場合、膜貫通領域内の変異のみをカウント。Falseの場合、領域外の変異のみをカウント
    """
    mutation_counts = defaultdict(lambda: {"Disease": 0, "Polymorphism": 0})
    is_filtered = allowed_acs is not None

    for ac, variants in data.items():
        if is_filtered and ac not in allowed_acs:
            continue
        
        if not isinstance(variants, list):
            continue
        
        # このACの膜貫通領域を取得
        transmembrane_regions = []
        if ac in transmembrane_data:
            regions = transmembrane_data[ac]
            if isinstance(regions, list):
                transmembrane_regions = regions
        
        for variant_info in variants:
            if not isinstance(variant_info, dict):
                continue
            
            clinical_sig = variant_info.get("Clinical_significance")
            if clinical_sig not in ["Disease", "Polymorphism"]:
                continue

            # 位置情報を取得
            position = variant_info.get("Position")
            if position is None:
                continue
            
            # 膜貫通領域内/外の判定
            is_in_tm = is_position_in_transmembrane_regions(position, transmembrane_regions)
            
            # フィルタリング条件に合致しない場合はスキップ
            if in_transmembrane and not is_in_tm:
                continue
            if not in_transmembrane and is_in_tm:
                continue

            variant = variant_info.get("Variant", {})
            if not isinstance(variant, dict):
                continue
            
            orig_aa_3 = variant.get("before")
            mut_aa_3 = variant.get("after")

            orig_aa_1 = AMINO_ACID_MAP.get(orig_aa_3)
            mut_aa_1 = AMINO_ACID_MAP.get(mut_aa_3)

            if not (orig_aa_1 and mut_aa_1):
                continue

            mutation_key = f"{orig_aa_1}->{mut_aa_1}"
            mutation_counts[mutation_key][clinical_sig] += 1
    
    return mutation_counts

def group_proteins_by_transmembrane_count(transmembrane_data):
    """
    膜貫通回数ごとにタンパク質を分類する
    
    Args:
        transmembrane_data: 膜貫通領域データ {AC: [{"start": int, "end": int}, ...]}
    
    Returns:
        dict: {膜貫通回数: [ACのリスト]}
    """
    groups = defaultdict(list)
    
    if not isinstance(transmembrane_data, dict):
        return groups
    
    for ac, regions in transmembrane_data.items():
        if not isinstance(regions, list):
            continue
        tm_count = len(regions)
        if 0 <= tm_count <= 15:
            groups[tm_count].append(ac)
    
    return groups

def analyze_transmembrane_heatmaps(clinvar_json_filepath, transmembrane_json_filepath, normalize=False, max_abs_value=None):
    """
    膜貫通領域に存在する変異と領域外に存在する変異のヒートマップを生成する
    """
    # JSONファイルを読み込む
    print(f"ClinVar JSONファイルを読み込んでいます: {clinvar_json_filepath}")
    try:
        with open(clinvar_json_filepath, 'r', encoding='utf-8') as f:
            clinvar_data = json.load(f)
    except FileNotFoundError:
        print(f"エラー: JSONファイルが見つかりません: {clinvar_json_filepath}")
        return
    except json.JSONDecodeError:
        print(f"エラー: JSONファイルの形式が正しくありません: {clinvar_json_filepath}")
        return
    
    print(f"膜貫通領域JSONファイルを読み込んでいます: {transmembrane_json_filepath}")
    try:
        with open(transmembrane_json_filepath, 'r', encoding='utf-8') as f:
            transmembrane_data = json.load(f)
    except FileNotFoundError:
        print(f"エラー: JSONファイルが見つかりません: {transmembrane_json_filepath}")
        return
    except json.JSONDecodeError:
        print(f"エラー: JSONファイルの形式が正しくありません: {transmembrane_json_filepath}")
        return
    
    # データ構造の検証
    if not isinstance(clinvar_data, dict):
        print(f"エラー: ClinVar JSONデータが辞書形式ではありません")
        return
    
    if not isinstance(transmembrane_data, dict):
        print(f"エラー: 膜貫通領域JSONデータが辞書形式ではありません")
        return
    
    # データベース全体の統計を計算 (P_all)
    print("\nデータベース全体の統計を計算中...")
    all_mutation_counts = defaultdict(lambda: {"Disease": 0, "Polymorphism": 0})
    for ac, variants in clinvar_data.items():
        if not isinstance(variants, list):
            continue
        
        for variant_info in variants:
            if not isinstance(variant_info, dict):
                continue
            
            clinical_sig = variant_info.get("Clinical_significance")
            if clinical_sig not in ["Disease", "Polymorphism"]:
                continue
            
            variant = variant_info.get("Variant", {})
            if not isinstance(variant, dict):
                continue
            
            orig_aa_3 = variant.get("before")
            mut_aa_3 = variant.get("after")
            
            orig_aa_1 = AMINO_ACID_MAP.get(orig_aa_3)
            mut_aa_1 = AMINO_ACID_MAP.get(mut_aa_3)
            
            if not (orig_aa_1 and mut_aa_1):
                continue
            
            mutation_key = f"{orig_aa_1}->{mut_aa_1}"
            all_mutation_counts[mutation_key][clinical_sig] += 1
    
    total_disease_all = sum(counts["Disease"] for counts in all_mutation_counts.values())
    total_polymorphism_all = sum(counts["Polymorphism"] for counts in all_mutation_counts.values())
    total_classified_all = total_disease_all + total_polymorphism_all
    p_all = total_classified_all > 0 and total_disease_all / total_classified_all or None
    
    p_all_str = f"{p_all:.4f}" if p_all is not None else "N/A"
    print(f"全体統計: Disease={total_disease_all}, Polymorphism={total_polymorphism_all}, P_all={p_all_str}")
    
    # 膜貫通回数ごとにタンパク質を分類
    print("\n膜貫通回数ごとにタンパク質を分類中...")
    tm_groups = group_proteins_by_transmembrane_count(transmembrane_data)
    
    print(f"膜貫通回数ごとのタンパク質数:")
    for tm_count in sorted(tm_groups.keys()):
        print(f"  {tm_count}回膜貫通型: {len(tm_groups[tm_count])}個のタンパク質")
    
    # ファイル名のベース部分を取得
    json_basename = os.path.splitext(os.path.basename(clinvar_json_filepath))[0]
    
    # 各膜貫通回数についてヒートマップを生成
    print("\nヒートマップを生成しています...")
    for tm_count in sorted(tm_groups.keys()):
        if tm_count < 0 or tm_count > 15:
            continue
        
        allowed_acs = set(tm_groups[tm_count])
        
        if not allowed_acs:
            print(f"  警告: {tm_count}回膜貫通型のタンパク質がありません。スキップします。")
            continue
        
        print(f"\n{tm_count}回膜貫通型の処理中...")
        print(f"  対象タンパク質数: {len(allowed_acs)}")
        
        # 膜貫通回数0の場合は、膜貫通領域がないので領域外の変異のみを処理
        if tm_count == 0:
            # 膜貫通領域外の変異をカウント（全ての変異が領域外）
            print(f"  膜貫通領域外の変異をカウント中...")
            try:
                tm_out_counts = count_mutations_from_data(
                    clinvar_data, 
                    transmembrane_data, 
                    allowed_acs=allowed_acs, 
                    in_transmembrane=False
                )
            except Exception as e:
                print(f"  エラー: 膜貫通領域外の変異をカウント中にエラーが発生しました: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            # 膜貫通領域外のヒートマップを生成
            title_out = f"Heatmap of Disease Ratio (膜貫通領域なし, Total Count >= 5)"
            filename_out = f"Heatmap_{json_basename}_TM0_out_transmembrane.pdf"
            if normalize:
                title_out = f"Heatmap of Normalized Score (膜貫通領域なし, Total Count >= 5)"
                filename_out = f"Heatmap_{json_basename}_TM0_out_transmembrane_normalized.pdf"
                if max_abs_value is not None:
                    filename_out = f"Heatmap_{json_basename}_TM0_out_transmembrane_normalized_{max_abs_value}.pdf"
            else:
                if max_abs_value is not None:
                    filename_out = f"Heatmap_{json_basename}_TM0_out_transmembrane_{max_abs_value}.pdf"
            
            try:
                plot_mutation_heatmap(
                    tm_out_counts, 
                    AMINO_ACIDS_ORDERED, 
                    title_out, 
                    filename_out, 
                    normalize=normalize, 
                    p_all=p_all, 
                    max_abs_value=max_abs_value
                )
            except Exception as e:
                print(f"  エラー: 膜貫通領域外のヒートマップ生成中にエラーが発生しました: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            # 統計情報を表示
            total_out = sum(c["Disease"] + c["Polymorphism"] for c in tm_out_counts.values())
            print(f"  変異数: {total_out}")
        
        else:
            # 膜貫通回数1-15の場合は、領域内と領域外の両方を処理
            # 膜貫通領域内の変異をカウント
            print(f"  膜貫通領域内の変異をカウント中...")
            try:
                tm_in_counts = count_mutations_from_data(
                    clinvar_data, 
                    transmembrane_data, 
                    allowed_acs=allowed_acs, 
                    in_transmembrane=True
                )
            except Exception as e:
                print(f"  エラー: 膜貫通領域内の変異をカウント中にエラーが発生しました: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            # 膜貫通領域外の変異をカウント
            print(f"  膜貫通領域外の変異をカウント中...")
            try:
                tm_out_counts = count_mutations_from_data(
                    clinvar_data, 
                    transmembrane_data, 
                    allowed_acs=allowed_acs, 
                    in_transmembrane=False
                )
            except Exception as e:
                print(f"  エラー: 膜貫通領域外の変異をカウント中にエラーが発生しました: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            # 膜貫通領域内のヒートマップを生成
            title_in = f"Heatmap of Disease Ratio ({tm_count}回膜貫通型, 膜貫通領域内, Total Count >= 5)"
            filename_in = f"Heatmap_{json_basename}_TM{tm_count}_in_transmembrane.pdf"
            if normalize:
                title_in = f"Heatmap of Normalized Score ({tm_count}回膜貫通型, 膜貫通領域内, Total Count >= 5)"
                filename_in = f"Heatmap_{json_basename}_TM{tm_count}_in_transmembrane_normalized.pdf"
                if max_abs_value is not None:
                    filename_in = f"Heatmap_{json_basename}_TM{tm_count}_in_transmembrane_normalized_{max_abs_value}.pdf"
            else:
                if max_abs_value is not None:
                    filename_in = f"Heatmap_{json_basename}_TM{tm_count}_in_transmembrane_{max_abs_value}.pdf"
            
            try:
                plot_mutation_heatmap(
                    tm_in_counts, 
                    AMINO_ACIDS_ORDERED, 
                    title_in, 
                    filename_in, 
                    normalize=normalize, 
                    p_all=p_all, 
                    max_abs_value=max_abs_value
                )
            except Exception as e:
                print(f"  エラー: 膜貫通領域内のヒートマップ生成中にエラーが発生しました: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            # 膜貫通領域外のヒートマップを生成
            title_out = f"Heatmap of Disease Ratio ({tm_count}回膜貫通型, 膜貫通領域外, Total Count >= 5)"
            filename_out = f"Heatmap_{json_basename}_TM{tm_count}_out_transmembrane.pdf"
            if normalize:
                title_out = f"Heatmap of Normalized Score ({tm_count}回膜貫通型, 膜貫通領域外, Total Count >= 5)"
                filename_out = f"Heatmap_{json_basename}_TM{tm_count}_out_transmembrane_normalized.pdf"
                if max_abs_value is not None:
                    filename_out = f"Heatmap_{json_basename}_TM{tm_count}_out_transmembrane_normalized_{max_abs_value}.pdf"
            else:
                if max_abs_value is not None:
                    filename_out = f"Heatmap_{json_basename}_TM{tm_count}_out_transmembrane_{max_abs_value}.pdf"
            
            try:
                plot_mutation_heatmap(
                    tm_out_counts, 
                    AMINO_ACIDS_ORDERED, 
                    title_out, 
                    filename_out, 
                    normalize=normalize, 
                    p_all=p_all, 
                    max_abs_value=max_abs_value
                )
            except Exception as e:
                print(f"  エラー: 膜貫通領域外のヒートマップ生成中にエラーが発生しました: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            # 統計情報を表示
            total_in = sum(c["Disease"] + c["Polymorphism"] for c in tm_in_counts.values())
            total_out = sum(c["Disease"] + c["Polymorphism"] for c in tm_out_counts.values())
            print(f"  膜貫通領域内の変異数: {total_in}")
            print(f"  膜貫通領域外の変異数: {total_out}")
    
    print("\n全てのヒートマップの生成が完了しました。")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="膜貫通領域に存在する変異と領域外に存在する変異のヒートマップを生成する"
    )
    parser.add_argument(
        "clinvar_json", 
        help="入力となるClinVar JSONファイルのパス"
    )
    parser.add_argument(
        "transmembrane_json", 
        help="膜貫通領域情報を含むJSONファイルのパス"
    )
    parser.add_argument(
        "--normalize", 
        "-n", 
        action='store_true', 
        help="規格化スコアを使用してヒートマップを生成する"
    )
    parser.add_argument(
        "--max_abs_value", 
        "-m", 
        type=float, 
        default=None, 
        help="規格化モードでのカラースケールの最大絶対値（例: 0.8で範囲-0.8から0.8）"
    )
    
    args = parser.parse_args()
    
    analyze_transmembrane_heatmaps(
        clinvar_json_filepath=args.clinvar_json,
        transmembrane_json_filepath=args.transmembrane_json,
        normalize=args.normalize,
        max_abs_value=args.max_abs_value
    )

