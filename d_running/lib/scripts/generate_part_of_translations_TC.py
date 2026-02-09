import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from lib.models import load_model_and_tokenizer_with_TC_layer
from lib.translate import (
    get_gold_function,
    preprocess_gold_function,
    translate_function,
)


def main(args):
    outs_path = args.outs_path
    tc_layer_path = args.tc_layer_path
    d_params_path = args.d_params_path
    originals_path = args.originals_path
    os.makedirs(outs_path, exist_ok=True)

    model, tok = load_model_and_tokenizer_with_TC_layer(
        os.path.join(args.model_path, "model"), tc_layer_path
    )

    # only generate translations that are not present yet
    files = os.listdir(d_params_path)
    already_existing_translations = os.listdir(outs_path)
    files = [f for f in files if f not in already_existing_translations]

    # limit to only files for this process
    part_size = int(len(files) / args.total_count)
    start = part_size * args.index
    end = part_size * (args.index + 1)
    if args.index + 1 == args.total_count:
        end = len(files)
    files = files[start:end]

    for file in files:
        try:
            original_path = os.path.join(originals_path, file.replace(".d", ".java"))
            with open(original_path, "r", encoding="utf8") as f:
                code = f.read()

            gold_function = get_gold_function(code)
            gold_function = preprocess_gold_function(gold_function)
            d_function = translate_function(model, tok, gold_function)
            d_function = d_function.replace("solution", "f_filled")  # rename function

            out_path = os.path.join(outs_path, file)
            with open(out_path, "w", encoding="utf8") as f:
                f.write(d_function)
        except Exception as e:
            print(f"{file} failed with {e}")


def parse_compound_defaults(args):
    if args.outs_path is None:
        args.outs_path = os.path.join(args.model_path, "translations")
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--total_count", type=int, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tc_layer_path", type=str, required=True)
    parser.add_argument("--outs_path", default=None)
    parser.add_argument("--d_params_path", default="data/d_with_params")
    parser.add_argument("--originals_path", default="path_to_geeks4geeks")

    main(parse_compound_defaults(parser.parse_args()))
