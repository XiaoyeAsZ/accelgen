#!/bin/bash
# Re-run only the 36 failed elementwise ops with modified problem.yaml
# (P folded from 4096 to 512, Q scaled up to preserve total computes)
set -e
cd /home/accelgen

DESIGNS="example_designs/example_designs"
TOP="$DESIGNS/top.yaml.jinja2"

run_one() {
  local arch="$1"
  local problem="$2"
  local outdir="$3"
  local opname="$4"

  echo "  [$opname] arch=$arch outdir=$outdir"
  python3 -c "
import timeloopfe.v4 as tl
spec = tl.Specification.from_yaml_files(
    '$TOP',
    jinja_parse_data={
        'architecture': '$arch',
        'problem': '$problem'
    }
)
tl.call_mapper(spec, output_dir='$outdir')
" 2>&1 | tail -2
}

total=0
fail=0

# gemma-7b elementwise ops on PPU_server (3 archs × 7 ops = 21)
for linear_arch in lego_server gemmini_os_server gemmini_ws_server; do
  for op in sqrt_0 divf_1 erf_2 addf_3 mulf_4 mulf_5 mulf_6; do
    base="baseline_test0412/$linear_arch/gemma-7b/ffn/prefill/b8s4096/$op"
    prob="/home/accelgen/$base/problem.yaml"
    total=$((total+1))
    echo "[$total/36] $linear_arch gemma-7b $op"
    if run_one "PPU_server" "$prob" "$base" "$op"; then
      echo "  ✓ OK"
    else
      echo "  ✗ FAILED"
      fail=$((fail+1))
    fi
  done
done

# llama3-8b elementwise ops on PPU_server (3 archs × 5 ops = 15)
for linear_arch in lego_server gemmini_os_server gemmini_ws_server; do
  for op in negf_0 addf_2 divf_3 mulf_4 mulf_5; do
    base="baseline_test0412/$linear_arch/llama3-8b/ffn/prefill/b8s4096/$op"
    prob="/home/accelgen/$base/problem.yaml"
    total=$((total+1))
    echo "[$total/36] $linear_arch llama3-8b $op"
    if run_one "PPU_server" "$prob" "$base" "$op"; then
      echo "  ✓ OK"
    else
      echo "  ✗ FAILED"
      fail=$((fail+1))
    fi
  done
done

echo ""
echo "================================"
echo "Done: $total runs, $fail failed"
echo "================================"
