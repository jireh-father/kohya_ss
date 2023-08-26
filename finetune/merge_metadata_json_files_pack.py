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

    caption_data_list = []
    for json_file in json_files:
        caption_data_list.append(json.loads(Path(json_file).read_text(encoding='utf-8')))

    metadata = defaultdict(dict)

    with open(args.in_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            file_id = os.path.splitext(data['file_name'])[0]
            pack_name = data['pack_name']
            pack_id = data['pack_id']
            captions = []
            for caption_data in caption_data_list:
                icons_caption = []
                for item in data['items']:
                    icon_id = os.path.splitext(item['file_name'])[0]
                    icon_text = item['text']

                    icons_caption.append(caption_data[icon_id]['caption'] + ", " + icon_text + ".")

                all_icon_caption = f"upper left is {icons_caption[0]} upper right is {icons_caption[1]} lower left is {icons_caption[2]} lower right is {icons_caption[3]}"

                full_caption = f"a pack of four icons, pack_name: {pack_name}, pack_id: {pack_id}. {all_icon_caption}"
                captions.append(full_caption)
            metadata[file_id]['caption'] = captions

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    # metadataを書き出して終わり
    print(f"writing metadata: {args.out_json}")
    Path(args.out_json).write_text(json.dumps(metadata, indent=2), encoding='utf-8')

    print("done!")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_json", type=str, help="metadata file to output / メタデータファイル書き出し先")
    parser.add_argument("--in_json_root", type=str)
    parser.add_argument("--in_jsons", type=str,
                        help="metadata file to input (if omitted and out_json exists, existing out_json is read) / 読み込むメタデータファイル（省略時、out_jsonが存在すればそれを読み込む）")
    parser.add_argument("--in_jsonl", type=str,
                        help="metadata file to input (if omitted and out_json exists, existing out_json is read) / 読み込むメタデータファイル（省略時、out_jsonが存在すればそれを読み込む）")

    return parser


if __name__ == '__main__':
    parser = setup_parser()

    args = parser.parse_args()
    main(args)
