#ifndef WRAPPER_H
#define WRAPPER_H

#include <string>

namespace mlir {
namespace accelgen {

class ParameterWrapper {
 public:
  ParameterWrapper();
  std::string parName;
  void* data;
};

class Int64Parameter : public ParameterWrapper {
 public:
  int64_t data;
  int64_t lowBound;
  int64_t upBound;
};

}  // namespace accelgen
}  // namespace mlir

#endif