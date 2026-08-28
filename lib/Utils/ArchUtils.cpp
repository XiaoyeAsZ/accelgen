#include "accelgen/Utils/ArchUtils.h"
#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>

namespace mlir::accelgen {

void ArchResource::load(llvm::StringRef cfgPath) {
  std::ifstream f(cfgPath.str());
  nlohmann::json config = nlohmann::json::parse(f);

  this->cycle = config["system"]["cycle"];
  this->bandwidth = config["dram"]["bandwidth"];
  this->dramEnergy = config["dram"]["energy"];
  this->nSramBank = config["sram"]["sram_bank"];
  this->sramWidth = config["sram"]["sram_width"];
  this->sramDepth = config["sram"]["sram_depth"];
  this->sramEnergy = config["sram"]["energy"];
  this->nDataNode = config["register"]["n"];

  auto comps = config["compute"];

  for (auto& [key, val] : comps.items()) {
    this->computeResource[key] = val["num"].get<int64_t>();
    this->computeEnergy[key] = val["energy"].get<double_t>();
  }
}

llvm::StringRef ArchResource::getArithName(llvm::StringRef name) {
  llvm::SmallVector<llvm::StringRef> tokens;
  name.split(tokens, "_");
  assert(tokens.size() > 0);
  return tokens[0];
}

ResourcePool::ResourcePool(llvm::ArrayRef<mlir::Operation*> resources)
    : _resourcesRef(resources), _currect(0) {}

llvm::ArrayRef<mlir::Operation*> ResourcePool::get(size_t nResource) {
  assert(_currect + nResource <= _resourcesRef.size());
  auto result = _resourcesRef.slice(_currect, nResource);
  _currect += nResource;
  return result;
}

void ResourcePool::reset() { _currect = 0; }

}  // namespace mlir::accelgen
