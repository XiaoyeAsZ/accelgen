import argparse
from torch_to_mlir import dump_to_mlir

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--action", type=str)
    parser.add_argument("--block", type=int)
    parser.add_argument("--layer", type=str)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--length", type=int)
    args = parser.parse_args()

    if args.model == "llama3-8b":
        from model.llama3_8b import build_model
    elif args.model == "qwen3-8b":
        from model.qwen3_8b import build_model
    elif args.model == "gemma-7b":
        from model.gemma_7b import build_model
    else:
        raise NotImplementedError()

    model, dummy_input = build_model(
        args.batch, args.length, args.action, args.block, args.layer
    )
    dump_to_mlir(
        f"./mlir/{args.model}-block{args.block}-{args.layer}-{args.action}-b{args.batch}s{args.length}.mlir",
        model,
        dummy_input,
    )
