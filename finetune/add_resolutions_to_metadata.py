import argparse
import json
from pathlib import Path
from typing import List
from tqdm import tqdm
import library.train_util as train_util
import os
from collections import defaultdict
import glob


def main(args):
    metadata = json.loads(Path(args.in_json).read_text(encoding='utf-8'))
    for k in metadata:
        metadata[k]['train_resolution'] = [args.resolution, args.resolution]

    json_files = glob.glob(args.in_jsons)
    if not json_files:
        json_files = args.in_jsons.split(',')
        if args.in_json_root is not None:
            json_files = [os.path.join(args.in_json_root, json_file) for json_file in json_files]

    metadata = {}
    for json_file in json_files:
        data = json.loads(Path(json_file).read_text(encoding='utf-8'))
        metadata.update(data)

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    # metadataを書き出して終わり
    print(f"writing metadata: {args.out_json}")
    Path(args.out_json).write_text(json.dumps(metadata, indent=2), encoding='utf-8')

    print("done!")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_json", type=str, help="metadata file to output / メタデータファイル書き出し先")
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--in_json", type=str,
                        help="metadata file to input (if omitted and out_json exists, existing out_json is read) / 読み込むメタデータファイル（省略時、out_jsonが存在すればそれを読み込む）")

    return parser


if __name__ == '__main__':
    parser = setup_parser()

    args = parser.parse_args()
    main(args)
