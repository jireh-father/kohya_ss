import argparse
import json
from pathlib import Path
from tqdm import tqdm
import os
import glob

def main(args):
  assert not args.recursive or (args.recursive and args.full_path), "recursive requires full_path / recursiveはfull_pathと同時に指定してください"

  caption_files = glob.glob(os.path.join(args.train_data_dir, f"*.{args.caption_extension}"))
  print(f"found {len(caption_files)} images.")

  if args.in_json is not None:
    print(f"loading existing metadata: {args.in_json}")
    metadata = json.loads(Path(args.in_json).read_text(encoding='utf-8'))
    print("captions for existing images will be overwritten / 既存の画像のキャプションは上書きされます")
  else:
    print("new metadata will be created / 新しいメタデータファイルが作成されます")
    metadata = {}

  print("merge caption texts to metadata json.")
  for caption_path in tqdm(caption_files):
    caption = open(caption_path, encoding='utf-8').read().strip()

    image_key = os.path.splitext(os.path.basename(caption_path))[0]
    if image_key not in metadata:
      metadata[image_key] = {}

    metadata[image_key]['caption'] = caption
    if args.debug:
      print(image_key, caption)

  # metadataを書き出して終わり
  print(f"writing metadata: {args.out_json}")
  Path(args.out_json).write_text(json.dumps(metadata, indent=2), encoding='utf-8')
  print("done!")


def setup_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser()
  parser.add_argument("train_data_dir", type=str, help="directory for train images / 学習画像データのディレクトリ")
  parser.add_argument("out_json", type=str, help="metadata file to output / メタデータファイル書き出し先")
  parser.add_argument("--in_json", type=str,
                      help="metadata file to input (if omitted and out_json exists, existing out_json is read) / 読み込むメタデータファイル（省略時、out_jsonが存在すればそれを読み込む）")
  parser.add_argument("--caption_extention", type=str, default=None,
                      help="extension of caption file (for backward compatibility) / 読み込むキャプションファイルの拡張子（スペルミスしていたのを残してあります）")
  parser.add_argument("--caption_extension", type=str, default=".caption", help="extension of caption file / 読み込むキャプションファイルの拡張子")
  parser.add_argument("--full_path", action="store_true",
                      help="use full path as image-key in metadata (supports multiple directories) / メタデータで画像キーをフルパスにする（複数の学習画像ディレクトリに対応）")
  parser.add_argument("--recursive", action="store_true",
                      help="recursively look for training tags in all child folders of train_data_dir / train_data_dirのすべての子フォルダにある学習タグを再帰的に探す")
  parser.add_argument("--debug", action="store_true", help="debug mode")

  return parser


if __name__ == '__main__':
  parser = setup_parser()

  args = parser.parse_args()

  # スペルミスしていたオプションを復元する
  if args.caption_extention is not None:
    args.caption_extension = args.caption_extention

  main(args)
