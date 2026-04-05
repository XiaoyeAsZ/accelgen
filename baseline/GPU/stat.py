if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--action", type=str)
    parser.add_argument("--block", type=int)
    parser.add_argument("--layer", type=str)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--length", type=int)
    parser.add_argument("--device", type=str, default="0")
    args = parser.parse_args()

    if args.model == "llama3-8b":
        from benchmark.model.llama3_8b import build_model
    elif args.model == "qwen3-8b":
        from benchmark.model.qwen3_8b import build_model
    elif args.model == "gemma-7b":
        from benchmark.model.gemma_7b import build_model
    else:
        raise NotImplementedError()

    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    import torch

    model, dummy_input = build_model(
        args.batch, args.length, args.action, args.block, args.layer
    )
    import time

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    dummy_output = model(*dummy_input)
    end.record()
    torch.cuda.synchronize()
    print(start.elapsed_time(end))
