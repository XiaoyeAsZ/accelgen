#!/usr/bin/env python3
"""Rerun only the unique failed Qwen3-MoE elementwise mappings."""

import argparse
import os

import timeloopfe.v4 as tl


PROBLEMS = {
    "attention-prefill-b8s2048/exp_26": dict(N=16384, C=5, P=64, Q=2048),
    "attention-prefill-b8s4096/mulf_22": dict(N=32768, C=1, P=64, Q=4096),
    "attention-prefill-b8s4096/addf_23": dict(N=32768, C=1, P=64, Q=4096),
    "attention-prefill-b8s4096/maximumf_reduction_24": dict(N=32768, C=12288, P=64, Q=1),
    "attention-prefill-b8s4096/subf_25": dict(N=32768, C=1, P=64, Q=4096),
    "attention-prefill-b8s4096/exp_26": dict(N=32768, C=5, P=64, Q=4096),
    "attention-prefill-b8s4096/divf_28": dict(N=32768, C=1, P=64, Q=4096),
    "ffn-prefill-b8s4096/exp_6": dict(N=64, C=5, P=2048, Q=1536),
}


def write_problem(path, dims):
    with open(path, "w", encoding="ascii") as f:
        f.write("{{include_text('/home/accelgen/example_designs/layer_shapes/problem_base.yaml')}}\n")
        f.write("problem:\n  <<<: *problem_base\n  instance:\n")
        for name in ("N", "C"):
            f.write(f"    {name}: {dims[name]}\n")
        f.write("    M: 1\n    R: 1\n    S: 1\n")
        for name in ("P", "Q"):
            f.write(f"    {name}: {dims[name]}\n")
        f.write("    G: 1\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        default="baseline_test_qwen3_moe/rerun_overrides/failed_ops150k",
    )
    args = parser.parse_args()
    top = "/home/accelgen/example_designs/example_designs/top.yaml.jinja2"
    for key, dims in PROBLEMS.items():
        output_dir = os.path.abspath(os.path.join(args.out_dir, key))
        os.makedirs(output_dir, exist_ok=True)
        problem_path = os.path.join(output_dir, "problem.yaml")
        write_problem(problem_path, dims)
        print(f"[{key}] N={dims['N']} C={dims['C']} P={dims['P']} Q={dims['Q']}", flush=True)
        spec = tl.Specification.from_yaml_files(
            top,
            jinja_parse_data={"architecture": "PPU_server", "problem": problem_path},
        )
        tl.call_mapper(spec, output_dir=output_dir)
    print(f"Completed {len(PROBLEMS)} unique failed mappings.")


if __name__ == "__main__":
    main()
