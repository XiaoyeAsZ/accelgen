./build/bin/accelgen-opt ./benchmark/mlir/llama3-8b.mlir \
-pass-pipeline="builtin.module(func.func(linalg-generalize-named-ops),mark-generic,fuse-generic)" \
-o ./benchmark/mlir/llama3-8b-generic-3k.mlir