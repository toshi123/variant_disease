import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List, Tuple
from scipy import stats

plt.rcParams['font.family'] = 'Hiragino Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42

# アミノ酸マッピング
AMINO_ACID_MAP = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D",
    "Cys": "C", "Gln": "Q", "Glu": "E", "Gly": "G",
    "His": "H", "Ile": "I", "Leu": "L", "Lys": "K",
    "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S",
    "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
    "Ter": "*", "Stp": "*"
}

# 設定: ACリストファイルとラベル
AC_LISTS_CONFIG = [
    ("ac_transmem_1pass.txt", "1pass"),
    ("ac_transmem_2pass.txt", "2pass"),
    ("ac_transmem_3pass.txt", "3pass"),
    ("ac_transmem_4pass.txt", "4pass"),
    ("ac_transmem_5pass.txt", "5pass"),
    ("ac_transmem_6pass.txt", "6pass"),
    ("ac_transmem_7pass.txt", "7pass"),
    ("ac_transmem_8pass.txt", "8pass"),
    ("ac_transmem_9pass.txt", "9pass"),
    ("ac_transmem_10pass.txt", "10pass"),
    ("ac_transmem_11pass.txt", "11pass"),
    ("ac_transmem_12pass.txt", "12pass"),
]

CLINVAR_JSON = "Clinvar_1ac_human_classfied.json"


def load_json(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_ac_list(path: str) -> List[str]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"警告: {path} が見つかりません。スキップします。")
        return []


def count_mutations(data: Dict[str, list], allowed_acs: set = None) -> Dict[str, Dict[str, int]]:
    """変異をカウント（Disease/Polymorphism別に）"""
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


def compute_log_scores(mutation_counts_all: Dict[str, Dict[str, int]], mutation_counts_subset: Dict[str, Dict[str, int]], min_count: int = 5) -> List[float]:
    """analyze_clinvar_json.pyと同じ正規化スコアを計算"""
    # 全体の疾患割合 P_all を計算
    total_disease_all = sum(counts["Disease"] for counts in mutation_counts_all.values())
    total_polymorphism_all = sum(counts["Polymorphism"] for counts in mutation_counts_all.values())
    total_classified_all = total_disease_all + total_polymorphism_all
    
    if total_classified_all == 0:
        return []
    
    p_all = total_disease_all / total_classified_all
    
    if p_all <= 0:
        return []
    
    scores = []
    for mutation_key, counts in mutation_counts_subset.items():
        pathogenic_count = counts.get("Disease", 0)
        benign_count = counts.get("Polymorphism", 0)
        total_classified = pathogenic_count + benign_count
        
        # 5つ未満の変異は除外
        if total_classified < min_count:
            continue
            
        # P_filtered を計算
        p_filtered = pathogenic_count / total_classified if total_classified > 0 else 0
        
        if p_filtered > 0:
            # 正規化スコア S_d = log(P_filtered / P_all)
            score = np.log(p_filtered / p_all)
            scores.append(score)
    
    return scores


def main():
    # ClinVarデータを読み込み
    print(f"ClinVarデータを読み込み中: {CLINVAR_JSON}")
    data = load_json(CLINVAR_JSON)
    
    # 全体の変異カウントを計算
    print("全体の変異カウントを計算中...")
    mutation_counts_all = count_mutations(data, allowed_acs=None)
    
    # 各ACリストに対してスコアを計算
    all_scores = []
    labels = []
    group_stats = []  # 各グループの詳細統計を保存
    
    for ac_file, label in AC_LISTS_CONFIG:
        print(f"処理中: {ac_file} ({label})")
        ac_list = load_ac_list(ac_file)
        if not ac_list:
            continue
            
        allowed_acs = set(ac_list)
        mutation_counts_subset = count_mutations(data, allowed_acs=allowed_acs)
        scores = compute_log_scores(mutation_counts_all, mutation_counts_subset, min_count=5)
        
        # グループの詳細統計を計算
        total_disease = sum(counts["Disease"] for counts in mutation_counts_subset.values())
        total_polymorphism = sum(counts["Polymorphism"] for counts in mutation_counts_subset.values())
        total_variants = total_disease + total_polymorphism
        
        group_stat = {
            "label": label,
            "num_acs": len(ac_list),
            "total_variants": total_variants,
            "disease_variants": total_disease,
            "polymorphism_variants": total_polymorphism
        }
        
        if scores:
            all_scores.append(scores)
            labels.append(label)
            group_stats.append(group_stat)
            print(f"  {len(scores)} 個の変異スコアを取得")
            print(f"  AC数: {len(ac_list)}, 総変異数: {total_variants}, Disease: {total_disease}, Polymorphism: {total_polymorphism}")
        else:
            print(f"  スコアが計算できませんでした")
    
    if not all_scores:
        print("エラー: 有効なスコアが計算できませんでした")
        return
    
    # ボックスプロットを作成
    print("ボックスプロットを作成中...")
    fig, ax = plt.subplots(figsize=(14, 8))
    
    bp = ax.boxplot(all_scores, labels=labels, patch_artist=True)
    
    # ボックスプロットの色を設定
    colors = plt.cm.Set3(np.linspace(0, 1, len(all_scores)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # 中央値ラインを黒に設定
    for median in bp['medians']:
        median.set_color('black')
    
    ax.set_xlabel('膜貫通回数')
    ax.set_ylabel('正規化スコア log(P_subset / P_all)')
    ax.set_title('膜貫通回数別の変異スコア分布')
    ax.grid(True, alpha=0.3)
    
    # 横軸ラベルを回転
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # 統計情報を表示
    print("\n=== 統計情報 ===")
    for i, (scores, label) in enumerate(zip(all_scores, labels)):
        print(f"{label}: N={len(scores)}, Mean={np.mean(scores):.3f}, Std={np.std(scores):.3f}")
    
    plt.tight_layout()
    
    # PDFとして保存
    output_file = "MutationScoresBoxplot_TransmemPasses.pdf"
    plt.savefig(output_file, format='pdf', bbox_inches='tight')
    print(f"\nボックスプロットを保存しました: {output_file}")
    
    # 統計サマリーをマークダウン形式で保存
    summary_file = "MutationScoresBoxplot_TransmemPasses_summary.md"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("# 膜貫通回数別変異スコア統計サマリー\n\n")
        
        # グループの詳細情報テーブル
        f.write("## グループ詳細情報\n\n")
        f.write("| 膜貫通回数 | AC数 | 総変異数 | Disease | Polymorphism | Disease割合(%) |\n")
        f.write("|:-----------|----:|--------:|--------:|-------------:|--------------:|\n")
        for stat in group_stats:
            disease_ratio = (stat["disease_variants"] / stat["total_variants"] * 100) if stat["total_variants"] > 0 else 0
            f.write(f"| {stat['label']} | {stat['num_acs']} | {stat['total_variants']} | "
                   f"{stat['disease_variants']} | {stat['polymorphism_variants']} | {disease_ratio:.2f} |\n")
        
        # 基本統計量のテーブル
        f.write("\n## スコア分布の基本統計量\n\n")
        f.write("| 膜貫通回数 | N | Mean | Std | Min | Max | Median |\n")
        f.write("|:-----------|--:|-----:|----:|----:|----:|-------:|\n")
        for scores, label in zip(all_scores, labels):
            f.write(f"| {label} | {len(scores)} | {np.mean(scores):.6f} | {np.std(scores):.6f} | "
                   f"{np.min(scores):.6f} | {np.max(scores):.6f} | {np.median(scores):.6f} |\n")
        
        # t検定の結果テーブル
        f.write("\n## t検定結果 (p値)\n\n")
        f.write("ウェルチのt検定（等分散を仮定しない）による総当たり比較\n\n")
        
        # ヘッダー行
        f.write("|")
        for label in labels:
            f.write(f" {label} |")
        f.write("\n")
        
        # 区切り行
        f.write("|")
        for _ in labels:
            f.write("-------:|")
        f.write("\n")
        
        # t検定結果の計算と出力
        for i, (scores_i, label_i) in enumerate(zip(all_scores, labels)):
            f.write(f"| **{label_i}** ")
            for j, (scores_j, label_j) in enumerate(zip(all_scores, labels)):
                if i == j:
                    f.write("| - ")
                elif i > j:
                    # 既に計算済みの場合は空欄
                    f.write("| ")
                else:
                    # t検定を実行
                    t_stat, p_value = stats.ttest_ind(scores_i, scores_j, equal_var=False)
                    if p_value < 0.001:
                        f.write(f"| <0.001 ")
                    else:
                        f.write(f"| {p_value:.3f} ")
            f.write("|\n")
        
        f.write("\n**注意:** 上三角行列のみ表示。対角線は自分自身との比較のため「-」、下三角は空欄。\n")
        f.write("\n**有意水準:** p < 0.05で統計的に有意な差があると判定。\n")
    
    print(f"統計サマリー（マークダウン）を保存しました: {summary_file}")


if __name__ == '__main__':
    main()
