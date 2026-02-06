./build/bin/accelgen-opt ./benchmark/mlir/llama3-8b-generic.mlir \
-pass-pipeline="builtin.module(func.func(kernel-schedule))" \
-o ./benchmark/mlir/llama3-8b-generic-scheduled.mlir \
2>&1 | tee ./log/log.txt