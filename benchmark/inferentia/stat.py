if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--action", type=str)
    parser.add_argument("--block", type=int)
    parser.add_argument("--layer", type=str)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--length", type=int)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--warm_up", type=int, default=10)
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    if args.model == "llama3-8b":
        config_path = Path(__file__).resolve().parent.parent / "model/llama3_8b"
        from benchmark.model.llama3_8b import build_model
    elif args.model == "qwen3-8b":
        config_path = Path(__file__).resolve().parent.parent / "model/qwen3_8b"
        from benchmark.model.qwen3_8b import build_model
    elif args.model == "qwen3-moe":
        config_path = (
            Path(__file__).resolve().parent.parent / "model/qwen3_moe/config.json"
        )
        from benchmark.model.qwen3_moe import build_model
    elif args.model == "gemma-7b":
        config_path = Path(__file__).resolve().parent.parent / "model/gemma_7b"
        from benchmark.model.gemma_7b import build_model
    else:
        raise NotImplementedError()

    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    import torch

    device = torch.device("cuda")

    model, dummy_input = build_model(
        args.batch,
        args.length,
        args.action,
        args.block,
        args.layer,
        device,
        local_path=config_path,
    )
    import time

    # warm-up (no timing, no sync needed strictly, but ok if you keep it)
    model.eval()
    with torch.no_grad():
        for _ in range(args.warm_up):
            _ = model(*dummy_input)

    torch.cuda.synchronize()

    # timing
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    times = []

    with torch.no_grad():
        for _ in range(args.iters):
            start.record()
            _ = model(*dummy_input)
            end.record()

            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))  # ms

    avg_time = sum(times) / len(times)

    print(f"Average latency over {args.iters} runs: {avg_time:.4f} ms")
    print(f"Std: {torch.tensor(times).std().item():.4f} ms")
