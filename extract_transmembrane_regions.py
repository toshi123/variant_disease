#!/usr/bin/env python3
"""
UniProt DATファイルから膜貫通領域（TRANSMEM）を抽出してJSONファイルを作成するスクリプト
"""

import json
import re
from typing import Dict, List

def parse_uniprot_dat(file_path: str) -> Dict[str, List[Dict[str, int]]]:
    """
    UniProt DATファイルを解析して膜貫通領域を抽出
    
    Args:
        file_path: UniProt DATファイルのパス
        
    Returns:
        膜貫通領域情報を含む辞書 {UniProt_ID: [{"start": int, "end": int}, ...]}
    """
    result = {}
    current_ac = None
    current_transmems = []
    
    # TRANSMEMの範囲を抽出する正規表現
    transmem_pattern = re.compile(r'FT\s+TRANSMEM\s+(\d+)\.\.(\d+)')
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # AC行（UniProt ID）を取得
            if line.startswith('AC   '):
                # AC行からUniProt IDを抽出（セミコロンで区切られた最初のID）
                ac_match = re.search(r'AC\s+([A-Z0-9]+)', line)
                if ac_match:
                    # 既存のエントリを保存
                    if current_ac:
                        result[current_ac] = current_transmems if current_transmems else []
                    current_ac = ac_match.group(1)
                    current_transmems = []
            
            # FT行でTRANSMEMを検索
            elif line.startswith('FT   TRANSMEM'):
                match = transmem_pattern.match(line)
                if match:
                    start = int(match.group(1))
                    end = int(match.group(2))
                    current_transmems.append({"start": start, "end": end})
            
            # エントリ終了（//）で現在のエントリを保存
            elif line.startswith('//'):
                if current_ac:
                    result[current_ac] = current_transmems if current_transmems else []
                    current_ac = None
                    current_transmems = []
        
        # ファイル終了時に最後のエントリを保存
        if current_ac:
            result[current_ac] = current_transmems if current_transmems else []
    
    return result


def main():
    input_file = 'uniprot_sprot_human.dat'
    output_file = 'transmembrane_regions.json'
    
    print(f"UniProt DATファイルを解析中: {input_file}")
    transmembrane_data = parse_uniprot_dat(input_file)
    
    print(f"膜貫通領域が見つかったタンパク質数: {sum(1 for v in transmembrane_data.values() if v)}")
    print(f"総タンパク質数: {len(transmembrane_data)}")
    
    # JSONファイルに保存
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(transmembrane_data, f, indent=4, ensure_ascii=False)
    
    print(f"結果を {output_file} に保存しました")


if __name__ == '__main__':
    main()

