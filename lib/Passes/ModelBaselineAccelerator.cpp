#include "accelgen/Passes/AccelgenPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include <assert.h>
#include <cstdlib>
#include <fstream>
#include <queue>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#include "accelgen/Utils/AffineMapUtils.h"
#include "accelgen/Utils/DebugUtils.h"
#include "accelgen/Utils/OperationUtils.h"

namespace mlir::accelgen {
#define GEN_PASS_DEF_MODELBASELINEACCELERATOR
#include "accelgen/Passes/AccelgenPasses.h.inc"

namespace {

// ============================================================================
// Data structure to hold per-operator timeloop results
// ============================================================================
struct TimeloopResult {
  std::string opName;
  std::string ssaName;  // MLIR SSA name, e.g. "%17", "%44:2"
  std::string opType;  // "linear" or "elementwise"
  std::string archName; // architecture used, e.g. "simba_like" or "PPU"
  int64_t cycles = 0;
  double energyUJ = 0.0;
  int64_t computes = 0;
  double utilization = 0.0;
  int64_t dramAccesses = 0;   // DRAM total scalar accesses
  int64_t sramAccesses = 0;   // On-chip (all non-DRAM buffers) total scalar accesses
  bool success = false;
};

// ============================================================================
// Helper: get SSA result name from an operation (e.g. "%17", "%44:2")
// ============================================================================
static std::string getSSAName(mlir::Operation* op, mlir::AsmState& asmState) {
  if (op->getNumResults() == 0) return "";
  std::string name;
  llvm::raw_string_ostream os(name);
  op->getResult(0).printAsOperand(os, asmState);
  os.flush();
  // Strip "#0" suffix for multi-result ops, show as "%44:2" instead
  if (op->getNumResults() > 1) {
    auto pos = name.find('#');
    if (pos != std::string::npos)
      name = name.substr(0, pos) + ":" + std::to_string(op->getNumResults());
  }
  return name;
};

// ============================================================================
// Helper: write a string to a file
// ============================================================================
static bool writeFile(const std::string& path, const std::string& content) {
  std::ofstream ofs(path);
  if (!ofs.is_open()) return false;
  ofs << content;
  ofs.close();
  return true;
}

// ============================================================================
// Helper: read entire file into a string
// ============================================================================
static std::string readFile(const std::string& path) {
  std::ifstream ifs(path);
  if (!ifs.is_open()) return "";
  std::stringstream ss;
  ss << ifs.rdbuf();
  return ss.str();
}

// ============================================================================
// Generate problem.yaml for a batch matmul (uses problem_base.yaml anchor)
//   GEMM: Out[B, M_gemm, N_gemm] = In[B, M_gemm, K] * W[B, K, N_gemm]
//   Conv mapping: M->N_gemm, C->K, P->M_gemm, N->batch, R=S=Q=1
// ============================================================================
static std::string generateBatchMatmulProblemYaml(
    const std::string& problemBasePath, int64_t batch, int64_t M_gemm,
    int64_t K, int64_t N_gemm) {
  std::ostringstream oss;
  oss << "{{include_text('" << problemBasePath << "')}}\n";
  oss << "problem:\n";
  oss << "  <<<: *problem_base\n";
  oss << "  instance:\n";
  oss << "    N: " << batch << "\n";
  oss << "    C: " << K << "\n";
  oss << "    M: " << N_gemm << "\n";
  oss << "    R: 1\n";
  oss << "    S: 1\n";
  oss << "    P: " << M_gemm << "\n";
  oss << "    Q: 1\n";
  oss << "    G: 1\n";
  return oss.str();
}

// ============================================================================
// Generate problem.yaml with explicit Conv2D dimensions
// ============================================================================
static std::string generateProblemYaml(
    const std::string& problemBasePath,
    int64_t N, int64_t C, int64_t P, int64_t Q) {
  std::ostringstream oss;
  oss << "{{include_text('" << problemBasePath << "')}}\n";
  oss << "problem:\n";
  oss << "  <<<: *problem_base\n";
  oss << "  instance:\n";
  oss << "    N: " << N << "\n";
  oss << "    C: " << C << "\n";
  oss << "    M: 1\n";
  oss << "    R: 1\n";
  oss << "    S: 1\n";
  oss << "    P: " << P << "\n";
  oss << "    Q: " << Q << "\n";
  oss << "    G: 1\n";
  return oss.str();
}

// ============================================================================
// Parse timeloop-mapper.stats.txt for key metrics
// ============================================================================
static TimeloopResult parseTimeloopStats(const std::string& statsPath,
                                         const std::string& opName,
                                         const std::string& opType) {
  TimeloopResult result;
  result.opName = opName;
  result.opType = opType;

  std::string content = readFile(statsPath);
  if (content.empty()) return result;

  result.success = true;

  // Parse Cycles (last occurrence = summary section)
  std::regex cyclesRe(R"(Cycles:\s+(\d+))");
  std::smatch match;
  std::string::const_iterator searchStart = content.cbegin();
  std::string lastCycles;
  while (std::regex_search(searchStart, content.cend(), match, cyclesRe)) {
    lastCycles = match[1].str();
    searchStart = match.suffix().first;
  }
  if (!lastCycles.empty()) result.cycles = std::stoll(lastCycles);

  // Parse Energy
  std::regex energyRe(R"(Energy:\s+([\d.eE+\-]+)\s*uJ)");
  if (std::regex_search(content, match, energyRe))
    result.energyUJ = std::stod(match[1].str());

  // Parse Computes
  std::regex computesRe(R"(Computes\s*=\s*(\d+))");
  if (std::regex_search(content, match, computesRe))
    result.computes = std::stoll(match[1].str());

  // Parse Utilization
  std::regex utilRe(R"(Utilization:\s+([\d.]+)%)");
  if (std::regex_search(content, match, utilRe))
    result.utilization = std::stod(match[1].str());

  // Parse memory accesses from the summary section at the bottom.
  // The summary section has lines like:
  //   === DRAM ===
  //       Total scalar accesses                   : 554147
  //   === GlobalBuffer ===
  //       Total scalar accesses                   : 2329015
  // We sum DRAM separately; everything else (on-chip) as SRAM.
  {
    // Find the summary section — it starts after the "Networks" or
    // "Operational Intensity Stats" header and has "=== XXX ===" + accesses.
    // We look for the LAST occurrence of each buffer's summary block.
    std::regex bufferRe(R"(=== (\w+) ===\s*\n\s*Total scalar accesses\s*:\s*(\d+))");
    auto summaryStart = content.cbegin();
    // Find "Operational Intensity Stats" to start from summary section
    auto opIntPos = content.find("Operational Intensity Stats");
    if (opIntPos != std::string::npos)
      summaryStart = content.cbegin() + opIntPos;
    while (std::regex_search(summaryStart, content.cend(), match, bufferRe)) {
      std::string bufName = match[1].str();
      int64_t accesses = std::stoll(match[2].str());
      if (bufName == "DRAM") {
        result.dramAccesses = accesses;
      } else if (bufName != "mac") {
        result.sramAccesses += accesses;
      }
      summaryStart = match.suffix().first;
    }
  }

  return result;
}

// ============================================================================
// Run timeloop via pytimeloop, reusing the existing example_designs framework
//   archTarget:  e.g. "simba_like" or "PPU"
//   exampleDesignsDir: root path containing example_designs/ and layer_shapes/
// ============================================================================
static TimeloopResult runTimeloop(const std::string& exampleDesignsDir,
                                  const std::string& archTarget,
                                  const std::string& problemYamlContent,
                                  const std::string& outputDir,
                                  const std::string& opName,
                                  const std::string& opType) {
  llvm::sys::fs::create_directories(outputDir);

  // Write generated problem yaml
  std::string problemPath = outputDir + "/problem.yaml";
  writeFile(problemPath, problemYamlContent);

  // Convert to absolute path (Jinja2 resolves relative to top.yaml.jinja2)
  llvm::SmallString<256> absProblemPath;
  llvm::sys::fs::real_path(problemPath, absProblemPath);
  std::string problemPathAbs = absProblemPath.str().str();

  // top.yaml.jinja2 orchestrates arch + mapper + components via Jinja2
  std::string topJinja =
      exampleDesignsDir + "/example_designs/top.yaml.jinja2";

  // Call pytimeloop Python API (same as run_example_designs.py)
  std::ostringstream pycmd;
  pycmd << "python3 -c \""
        << "import timeloopfe.v4 as tl; "
        << "spec = tl.Specification.from_yaml_files("
        << "'" << topJinja << "', "
        << "jinja_parse_data={'architecture': '" << archTarget << "', "
        << "'problem': '" << problemPathAbs << "'}); "
        << "tl.call_mapper(spec, output_dir='" << outputDir << "', "
        << "dump_intermediate_to='" << outputDir << "')"
        << "\" 2>&1";

  llvm::errs() << "=== Running timeloop for: " << opName << " (" << opType
               << ") ===\n";
  llvm::errs() << "  Arch: " << archTarget << "\n";
  llvm::errs() << "  Problem: " << problemPathAbs << "\n";
  llvm::errs() << "  Output: " << outputDir << "\n";

  int ret = std::system(pycmd.str().c_str());

  std::string statsPath = outputDir + "/timeloop-mapper.stats.txt";
  if (ret != 0 || !llvm::sys::fs::exists(statsPath)) {
    llvm::errs() << "  [ERROR] timeloop mapper failed for " << opName << "\n";
    TimeloopResult fail;
    fail.opName = opName;
    fail.opType = opType;
    fail.success = false;
    return fail;
  }

  auto result = parseTimeloopStats(statsPath, opName, opType);
  llvm::errs() << "  Cycles: " << result.cycles
               << "  Energy: " << result.energyUJ << " uJ"
               << "  Computes: " << result.computes
               << "  Utilization: " << result.utilization << "%\n";
  return result;
}

// ============================================================================
// Count arith ops in a generic op body
// ============================================================================
static int64_t countOpsInBody(linalg::GenericOp genericOp) {
  int64_t count = 0;
  for (auto& op : genericOp.getBody()->getOperations()) {
    if (mlir::isa<linalg::YieldOp>(op)) continue;
    count++;
  }
  return std::max(count, (int64_t)1);
}

// ============================================================================
// Compute total element count from a tensor shape
// ============================================================================
static int64_t getTotalElements(llvm::ArrayRef<int64_t> shape) {
  int64_t total = 1;
  for (auto dim : shape) total *= dim;
  return total;
}

// ============================================================================
// Check if a generic op has any broadcast input (input shape != output shape)
// ============================================================================
static bool hasBroadcastInput(linalg::GenericOp genericOp) {
  auto outShape = getOperandShape(genericOp.getOutputs()[0]);
  for (auto input : genericOp.getInputs()) {
    auto inShape = getOperandShape(input);
    if (inShape.size() != outShape.size()) return true;
    for (size_t i = 0; i < inShape.size(); i++) {
      if (inShape[i] != outShape[i]) return true;
    }
  }
  return false;
}

// ============================================================================
// Get the compute cost (C value) for a single MLIR operation.
// First tries to read from nonlin_op/{shortname}.yaml; if not found,
// falls back to hardcoded defaults.
// ============================================================================
static int64_t getOpComputeCost(llvm::StringRef opName,
                                const std::string& nonlinOpDir) {
  // Try reading C from per-op YAML file
  std::string shortName = opName.substr(opName.rfind('.') + 1).str();
  std::string yamlPath = nonlinOpDir + "/" + shortName + ".yaml";
  std::string content = readFile(yamlPath);
  if (!content.empty()) {
    std::regex cRe(R"(C:\s+(\d+))");
    std::smatch match;
    if (std::regex_search(content, match, cRe))
      return std::stoll(match[1].str());
  }

  // Fallback: hardcoded defaults
  if (opName == "math.exp")   return 5;
  if (opName == "math.erf")   return 5;
  if (opName == "math.log")   return 5;
  if (opName == "math.tanh")  return 5;
  if (opName == "math.rsqrt") return 3;
  if (opName == "math.sqrt")  return 3;
  if (opName == "math.cos")   return 5;
  if (opName == "math.sin")   return 5;
  return 1;
}

// ============================================================================
// Extract short name from a full MLIR op name (e.g. "arith.negf" → "negf")
// ============================================================================
static std::string getOpShortName(llvm::StringRef fullName) {
  auto dotPos = fullName.rfind('.');
  if (dotPos != llvm::StringRef::npos)
    return fullName.substr(dotPos + 1).str();
  return fullName.str();
}

// ============================================================================
// The main pass
// ============================================================================
class ModelBaselineAccelerator
    : public impl::ModelBaselineAcceleratorBase<ModelBaselineAccelerator> {
 public:
  using impl::ModelBaselineAcceleratorBase<
      ModelBaselineAccelerator>::ModelBaselineAcceleratorBase;

  void runOnOperation() final {
    auto module = getOperation();

    // configPath = path to the example_designs/ root directory
    //   e.g. /home/accelgen/example_designs
    // It contains: example_designs/top.yaml.jinja2, example_designs/simba_like/arch.yaml,
    //              example_designs/PPU/arch.yaml, layer_shapes/problem_base.yaml, etc.
    std::string exampleDesignsDir = configPath.empty()
                                        ? "example_designs"
                                        : configPath.getValue();

    // acceleratorName = the linear architecture target name (default: simba_like)
    std::string linearArchName = acceleratorName.empty()
                                     ? "simba_like"
                                     : acceleratorName.getValue();
    std::string nonlinearArchName = ppuName.empty()
                                       ? "PPU"
                                       : ppuName.getValue();

    std::string mlirFileName = mlirFile.empty() ? "(unknown)" : mlirFile.getValue();

    // Extract stem from mlir file path: "benchmark/mlir/llama3-8b-xxx.mlir" → "llama3-8b-xxx"
    std::string mlirStem = "unknown";
    if (!mlirFile.empty()) {
      llvm::StringRef fn = llvm::sys::path::stem(mlirFile.getValue());
      mlirStem = fn.str();
    }

    std::string outputBase = "baseline_test/" + linearArchName +
                             "/" + mlirStem;

    // problem_base.yaml provides the YAML anchor *problem_base
    std::string problemBasePath =
        exampleDesignsDir + "/layer_shapes/problem_base.yaml";

    // Per-op nonlinear YAML files (negf.yaml, exp.yaml, etc.)
    std::string nonlinOpDir =
        exampleDesignsDir + "/layer_shapes/nonlin_op";

    llvm::errs() << "\n[ModelBaseline] Config:\n"
                 << "  MLIR file:           " << mlirFileName << "\n"
                 << "  example_designs dir: " << exampleDesignsDir << "\n"
                 << "  linear arch:         " << linearArchName << "\n"
                 << "  nonlinear arch:      " << nonlinearArchName << "\n"
                 << "  problem_base:        " << problemBasePath << "\n"
                 << "  output base:         " << outputBase << "\n\n";

    int batchMatmulIdx = 0;
    int genericIdx = 0;
    std::vector<TimeloopResult> results;
    mlir::AsmState asmState(module);

    module.walk([&](mlir::Operation* op) {
      llvm::TypeSwitch<mlir::Operation*>(op)
          // ---- Linear: batch_matmul ----
          .Case<linalg::BatchMatmulOp>(
              [&](linalg::BatchMatmulOp batchMatmulOp) {
                auto inputA = batchMatmulOp.getInputs()[0];
                auto inputB = batchMatmulOp.getInputs()[1];
                auto shapeA = getOperandShape(inputA);
                auto shapeB = getOperandShape(inputB);

                int64_t batch = shapeA[0];
                int64_t M_gemm = shapeA[1];
                int64_t K = shapeA[2];
                int64_t N_gemm = shapeB[2];

                std::string opName =
                    "batch_matmul_" + std::to_string(batchMatmulIdx++);

                llvm::errs() << "\n[ModelBaseline] Found " << opName
                             << ": batch=" << batch << " M=" << M_gemm
                             << " K=" << K << " N=" << N_gemm << "\n";

                std::string problemYaml = generateBatchMatmulProblemYaml(
                    problemBasePath, batch, M_gemm, K, N_gemm);
                std::string outDir = outputBase + "/" + opName;

                auto result = runTimeloop(exampleDesignsDir, linearArchName,
                                          problemYaml, outDir, opName,
                                          "linear");
                result.ssaName = getSSAName(batchMatmulOp, asmState);
                result.archName = linearArchName;
                results.push_back(result);
              })
          // ---- Nonlinear: element-wise generic ops ----
          .Case<linalg::GenericOp>([&](linalg::GenericOp genericOp) {
            // Skip memory-only transformations
            if (genericOp->hasAttr("accelgen.memory_transformation"))
              return;

            // Count body ops, detect type-cast-only ops
            int64_t bodyOpCount = 0;
            bool isTypeCastOnly = true;
            for (auto& bodyOp : genericOp.getBody()->getOperations()) {
              if (mlir::isa<linalg::YieldOp>(bodyOp)) continue;
              bodyOpCount++;
              auto name = bodyOp.getName().getStringRef();
              if (name != "arith.extf" && name != "arith.truncf")
                isTypeCastOnly = false;
            }
            // Skip broadcasts (empty body) and pure type casts
            if (bodyOpCount == 0) return;
            if (isTypeCastOnly) {
              llvm::errs() << "\n[ModelBaseline] Skipping type cast op\n";
              return;
            }

            // Check for reduction
            bool hasReduction = false;
            for (auto it : genericOp.getIteratorTypesArray())
              if (it == mlir::utils::IteratorType::reduction)
                hasReduction = true;

            auto outputs = genericOp.getOutputs();
            if (outputs.empty()) return;
            auto outShape = getOperandShape(outputs[0]);
            if (outShape.empty()) return;

            if (hasReduction) {
              // ---- Reduction ops ----
              // C = reduced dimension size, scaled by number of body ops
              auto inShape = getOperandShape(genericOp.getInputs()[0]);
              int64_t reducedDim = 1;
              if (inShape.size() > outShape.size()) {
                reducedDim = inShape.back();
              } else {
                for (size_t i = 0; i < inShape.size(); i++) {
                  if (inShape[i] > outShape[i]) { reducedDim = inShape[i]; break; }
                }
              }

              // Count non-trivial body ops per reduction step
              int64_t opsPerStep = 0;
              std::string firstName;
              for (auto& bodyOp : genericOp.getBody()->getOperations()) {
                if (mlir::isa<linalg::YieldOp>(bodyOp)) continue;
                if (mlir::isa<linalg::IndexOp>(bodyOp)) continue;
                auto name = bodyOp.getName().getStringRef();
                if (name == "arith.extf" || name == "arith.truncf" ||
                    name == "arith.index_cast" || name == "arith.constant")
                  continue;
                opsPerStep++;
                if (firstName.empty()) firstName = getOpShortName(name);
              }
              opsPerStep = std::max(opsPerStep, (int64_t)1);

              int64_t C = reducedDim * opsPerStep;
              int64_t N = 1, P = 1, Q = 1;
              if (outShape.size() >= 2) {
                P = outShape[1];
                N = outShape[0];
                for (size_t i = 2; i < outShape.size(); i++)
                  N *= outShape[i];
              } else if (outShape.size() == 1) {
                N = outShape[0];
              }

              std::string opName =
                  firstName + "_reduction_" + std::to_string(genericIdx++);

              llvm::errs() << "\n[ModelBaseline] Found " << opName
                           << " (reduction, " << opsPerStep << " ops/step)"
                           << "\n  shape=[";
              for (size_t i = 0; i < outShape.size(); i++) {
                if (i > 0) llvm::errs() << "x";
                llvm::errs() << outShape[i];
              }
              llvm::errs() << "] → N=" << N << " C=" << C
                           << " P=" << P << " Q=" << Q
                           << " reducedDim=" << reducedDim << "\n";

              std::string problemYaml =
                  generateProblemYaml(problemBasePath, N, C, P, Q);
              std::string outDir = outputBase + "/" + opName;

              auto result = runTimeloop(exampleDesignsDir, nonlinearArchName,
                                        problemYaml, outDir, opName,
                                        "elementwise");
              result.ssaName = getSSAName(genericOp, asmState);
              result.archName = nonlinearArchName;
              results.push_back(result);

            } else {
              // ---- Elementwise ops: evaluate each body op independently ----
              // Compute N/P/Q from output shape (same for all body ops)
              int64_t N = 1, P = 1, Q = 1;
              if (outShape.size() >= 4) {
                N = outShape[0];
                P = outShape[1];
                Q = outShape.back();
                for (size_t i = 2; i < outShape.size() - 1; i++)
                  N *= outShape[i];
              } else if (outShape.size() == 3) {
                N = outShape[0]; P = outShape[1]; Q = outShape[2];
              } else if (outShape.size() == 2) {
                N = outShape[0]; Q = outShape[1];
              } else {
                Q = outShape[0];
              }

              for (auto& bodyOp : genericOp.getBody()->getOperations()) {
                if (mlir::isa<linalg::YieldOp>(bodyOp)) continue;
                if (mlir::isa<linalg::IndexOp>(bodyOp)) continue;
                auto fullName = bodyOp.getName().getStringRef();
                // Skip type casts and constants (no compute cost)
                if (fullName == "arith.extf" || fullName == "arith.truncf" ||
                    fullName == "arith.constant")
                  continue;

                std::string shortName = getOpShortName(fullName);
                int64_t C = getOpComputeCost(fullName, nonlinOpDir);

                std::string opName =
                    shortName + "_" + std::to_string(genericIdx++);

                llvm::errs() << "\n[ModelBaseline] Found " << opName
                             << " (" << fullName.str() << ", C=" << C << ")"
                             << "\n  shape=[";
                for (size_t i = 0; i < outShape.size(); i++) {
                  if (i > 0) llvm::errs() << "x";
                  llvm::errs() << outShape[i];
                }
                llvm::errs() << "] → N=" << N << " C=" << C
                             << " P=" << P << " Q=" << Q << "\n";

                std::string problemYaml =
                    generateProblemYaml(problemBasePath, N, C, P, Q);
                std::string outDir = outputBase + "/" + opName;

                auto result = runTimeloop(exampleDesignsDir, nonlinearArchName,
                                          problemYaml, outDir, opName,
                                          "elementwise");
                result.ssaName = getSSAName(genericOp, asmState);
                result.archName = nonlinearArchName;
                results.push_back(result);
              }
            }
          })
          .Case<linalg::TransposeOp>([&](linalg::TransposeOp) {
            llvm::errs() << "\n[ModelBaseline] Skipping transpose op "
                         << "(memory-only, no compute)\n";
          })
          .Case<linalg::FillOp>([&](linalg::FillOp) {})
          .Default([](mlir::Operation*) {});
    });

    // ========================================================================
    // Print summary
    // ========================================================================
    llvm::errs() << "\n";
    llvm::errs()
        << "================================================================\n";
    llvm::errs()
        << "  ModelBaselineAccelerator — Timeloop Results Summary\n";
    llvm::errs()
        << "  MLIR file:    " << mlirFileName << "\n";
    llvm::errs()
        << "  Linear arch:  " << linearArchName
        << "  |  Nonlinear arch: " << nonlinearArchName << "\n";
    llvm::errs()
        << "================================================================\n";

    int64_t totalCycles = 0;
    double totalEnergy = 0.0;
    int64_t totalComputes = 0;
    int successCount = 0;

    int64_t totalDramAcc = 0;
    int64_t totalSramAcc = 0;

    llvm::errs() << "SSA        Operator                       Arch        Type         "
                 << "      Cycles   Energy(uJ)     Computes    Util%"
                 << "     DRAM_Access   SRAM_Access\n";
    llvm::errs() << std::string(145, '-') << "\n";

    for (auto& r : results) {
      if (r.success) {
        llvm::errs() << r.ssaName;
        for (size_t pad = r.ssaName.size(); pad < 11; pad++) llvm::errs() << ' ';
        llvm::errs() << r.opName;
        for (size_t pad = r.opName.size(); pad < 31; pad++) llvm::errs() << ' ';
        llvm::errs() << r.archName;
        for (size_t pad = r.archName.size(); pad < 12; pad++) llvm::errs() << ' ';
        llvm::errs() << r.opType;
        for (size_t pad = r.opType.size(); pad < 13; pad++) llvm::errs() << ' ';
        llvm::errs() << r.cycles << "\t" << r.energyUJ << "\t"
                     << r.computes << "\t" << r.utilization << "%"
                     << "\t" << r.dramAccesses << "\t" << r.sramAccesses << "\n";
        totalCycles += r.cycles;
        totalEnergy += r.energyUJ;
        totalComputes += r.computes;
        totalDramAcc += r.dramAccesses;
        totalSramAcc += r.sramAccesses;
        successCount++;
      } else {
        llvm::errs() << r.ssaName;
        for (size_t pad = r.ssaName.size(); pad < 11; pad++) llvm::errs() << ' ';
        llvm::errs() << r.opName;
        for (size_t pad = r.opName.size(); pad < 31; pad++) llvm::errs() << ' ';
        llvm::errs() << r.archName;
        for (size_t pad = r.archName.size(); pad < 12; pad++) llvm::errs() << ' ';
        llvm::errs() << r.opType;
        for (size_t pad = r.opType.size(); pad < 13; pad++) llvm::errs() << ' ';
        llvm::errs() << "FAILED\tFAILED\tFAILED\tN/A\tN/A\tN/A\n";
      }
    }

    llvm::errs() << std::string(145, '-') << "\n";
    llvm::errs() << "           TOTAL                                                    "
                 << totalCycles << "\t" << totalEnergy << "\t"
                 << totalComputes << "\t\t" << totalDramAcc << "\t" << totalSramAcc << "\n";
    llvm::errs()
        << "================================================================\n";
    llvm::errs() << "  Total operators: " << results.size()
                 << "  Succeeded: " << successCount
                 << "  Failed: " << (results.size() - successCount) << "\n";
    llvm::errs()
        << "================================================================\n";
  }
};

}  // namespace
}  // namespace mlir::accelgen
