import json
import argparse
import os

def check_location(position, feature_list):
    """
    指定された位置が、特徴領域のリスト内に存在するかを判定する。
    BINDING, TRANSIT のような複雑なリスト構造に対応。
    """
    if not feature_list:
        return False
    
    for item_dict in feature_list:
        # item_dict is like {"Mitochondrion": {"start":1, "end":20}}
        for key, range_dict in item_dict.items():
            if range_dict['start'] <= position <= range_dict['end']:
                return True
    return False

def check_simple_location(position, feature_list):
    """
    指定された位置が、単純な特徴領域のリスト内に存在するかを判定する。
    TRANSMEM, SIGNAL, PROPEP などに対応。
    """
    if not feature_list:
        return False
    
    return any(
        region['start'] <= position <= region['end']
        for region in feature_list
    )

def annotate_variants(clinvar_data, uniprot_features):
    """
    ClinVarの変異データにUniProtの特徴量情報を付与する。
    """
    # 出力用のデータ構造を準備（元のClinVarデータを直接変更する）
    annotated_data = clinvar_data.copy()

    for ac, variants in annotated_data.items():
        # UniProt側にACが存在しない場合の処理
        if ac not in uniprot_features:
            for variant in variants:
                variant["UniprotFeature"] = {
                    "TRANSMEM": "UNCLASSIFIED",
                    "TRANSIT": False,
                    "BINDING": False,
                    "ACT_SITE": False,
                    "SIGNAL": False,
                    "PROPEP": False,
                }
            continue # 次のACの処理へ
            
        features = uniprot_features[ac]
        
        # 各特徴量リストを取得（存在しない場合は空リスト）
        transmem_list = features.get("TRANSMEM", [])
        transit_list = features.get("TRANSIT", [])
        binding_list = features.get("BINDING", [])
        act_site_list = features.get("ACT_SITE", [])
        signal_list = features.get("SIGNAL", [])
        propep_list = features.get("PROPEP", [])
        
        has_transmem = bool(transmem_list)

        for variant in variants:
            try:
                position = int(variant["Position"])
            except (ValueError, KeyError):
                continue # Positionが不正な場合はスキップ

            # 1. TRANSMEMフラグの判定
            transmem_flag = "SOLUBLE_PROTEIN" # デフォルト
            if has_transmem:
                is_in_tmreg = check_simple_location(position, transmem_list)
                transmem_flag = "TMREG" if is_in_tmreg else "LOOP_DOMAIN"
            
            # 2. TRANSITフラグの判定
            transit_flag = check_location(position, transit_list)
            
            # 3. BINDINGフラグの判定
            binding_flag = check_location(position, binding_list)
            
            # 4. ACT_SITEフラグの判定
            act_site_flag = position in act_site_list
            
            # 5. SIGNALフラグの判定
            signal_flag = check_simple_location(position, signal_list)

            # 6. PROPEPフラグの判定
            propep_flag = check_simple_location(position, propep_list)

            # variant辞書に直接新しいキーを追加
            variant["UniprotFeature"] = {
                "TRANSMEM": transmem_flag,
                "TRANSIT": transit_flag,
                "BINDING": binding_flag,
                "ACT_SITE": act_site_flag,
                "SIGNAL": signal_flag,
                "PROPEP": propep_flag,
            }

    return annotated_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Annotate ClinVar variants with UniProt feature locations."
    )
    parser.add_argument(
        "clinvar_json",
        help="Path to the input ClinVar JSON file (e.g., Clinvar_1ac_human_classfied.json)"
    )
    parser.add_argument(
        "uniprot_features_json",
        help="Path to the UniProt features JSON file (e.g., uniprot_features.json)"
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Path to the output annotated JSON file. Defaults to a name based on the input."
    )
    
    args = parser.parse_args()

    # 出力ファイル名が指定されなかった場合のデフォルト名を作成
    if args.output is None:
        base, ext = os.path.splitext(os.path.basename(args.clinvar_json))
        args.output = f"{base}_annotated{ext}"

    # JSONファイルの読み込み
    try:
        with open(args.clinvar_json, 'r', encoding='utf-8') as f:
            clinvar_data = json.load(f)
        print(f"'{args.clinvar_json}' を正常に読み込みました。")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"エラー: '{args.clinvar_json}' の読み込みに失敗しました: {e}")
        exit(1)

    try:
        with open(args.uniprot_features_json, 'r', encoding='utf-8') as f:
            uniprot_features = json.load(f)
        print(f"'{args.uniprot_features_json}' を正常に読み込みました。")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"エラー: '{args.uniprot_features_json}' の読み込みに失敗しました: {e}")
        exit(1)

    # アノテーション処理の実行
    print("変異情報にアノテーションを追加しています...")
    annotated_result = annotate_variants(clinvar_data, uniprot_features)

    # 結果をJSONファイルに書き出し
    try:
        with open(args.output, 'w', encoding='utf-8') as f_out:
            json.dump(annotated_result, f_out, indent=4)
        print(f"アノテーション済みデータを '{args.output}' に保存しました。")
    except IOError as e:
        print(f"エラー: ファイルの書き込みに失敗しました: {e}")
        exit(1) 