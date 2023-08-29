import argparse
import json
from pathlib import Path
from typing import List
from tqdm import tqdm
import library.train_util as train_util
import os
from collections import defaultdict


def main(args):
    json_files = args.in_jsons.split(',')
    if args.in_json_root is not None:
        json_files = [os.path.join(args.in_json_root, json_file) for json_file in json_files]

    metadata = defaultdict(dict)
    num_jsons = 0
    is_jsonls = False
    for json_file in json_files:
        if json_file.endswith('.json'):
            num_jsons += 1
            data = json.loads(Path(json_file).read_text(encoding='utf-8'))
            for icon_id in data:
                for data_key in data[icon_id]:
                    if data_key not in metadata[icon_id]:
                        metadata[icon_id][data_key] = []
                    caption = data[icon_id][data_key]
                    if not caption.startswith("a icon of"):
                        caption = f"a icon of {caption}"
                    metadata[icon_id][data_key].append(caption)
        elif json_file.endswith('.jsonl'):
            is_jsonls = True
            with open(json_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    icon_id = os.path.splitext(data['file_name'])[0]
                    tags = data['text']
                    if 'name: ' in tags:
                        tags = tags.replace('name: ', '')
                    if 'style: ' in tags:
                        tags = tags.replace('style: ', '')
                    if 'color: ' in tags:
                        tags = tags.replace('color: ', '')
                    data_key = 'tags'
                    metadata[icon_id][data_key] = tags

    if args.skip_not_all:
        print("skip not all")
        metadata = {icon_id: data for icon_id, data in metadata.items() if 'caption' in data and len(data['caption']) == num_jsons and (not is_jsonls or 'tags' in data)}

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    # metadataを書き出して終わり
    print(f"writing metadata: {args.out_json}")
    Path(args.out_json).write_text(json.dumps(metadata, indent=2), encoding='utf-8')

    print("done!")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_json", type=str, help="metadata file to output / メタデータファイル書き出し先")
    parser.add_argument("--in_json_root", type=str)
    parser.add_argument("--skip_not_all", action='store_true', default=False)
    parser.add_argument("--in_jsons", type=str,
                        help="metadata file to input (if omitted and out_json exists, existing out_json is read) / 読み込むメタデータファイル（省略時、out_jsonが存在すればそれを読み込む）")

    return parser


if __name__ == '__main__':
    parser = setup_parser()

    args = parser.parse_args()
    main(args)
