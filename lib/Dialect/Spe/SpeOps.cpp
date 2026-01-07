#include "accelgen/Dialect/Spe/SpeOps.h"

#include "accelgen/Dialect/Spe/SpeDialect.h"

#define GET_OP_CLASSES
#include "accelgen/Dialect/Spe/SpeOps.cpp.inc"

void mlir::spe::MacOp::build(::mlir::OpBuilder &odsBuilder,
                             ::mlir::OperationState &odsState,
                             ValueRange operands,
                             ArrayRef<NamedAttribute> attrs) {
  odsState.addOperands(operands);
  odsState.addAttributes(attrs);
  assert(operands.size() == 3 && "Mac Op accept 3 operands");
  odsState.addTypes(operands[2].getType());
}

void mlir::spe::ExpOp::build(::mlir::OpBuilder &odsBuilder,
                             ::mlir::OperationState &odsState,
                             ValueRange operands,
                             ArrayRef<NamedAttribute> attrs) {
  odsState.addOperands(operands);
  odsState.addAttributes(attrs);
  assert(operands.size() == 1 && "Exp Op accept 1 operands");
  odsState.addTypes(operands[0].getType());
}

void mlir::spe::InvOp::build(::mlir::OpBuilder &odsBuilder,
                             ::mlir::OperationState &odsState,
                             ValueRange operands,
                             ArrayRef<NamedAttribute> attrs) {
  odsState.addOperands(operands);
  odsState.addAttributes(attrs);
  assert(operands.size() == 1 && "Inv Op accept 1 operands");
  odsState.addTypes(operands[0].getType());
}

void mlir::spe::NegOp::build(::mlir::OpBuilder &odsBuilder,
                             ::mlir::OperationState &odsState,
                             ValueRange operands,
                             ArrayRef<NamedAttribute> attrs) {
  odsState.addOperands(operands);
  odsState.addAttributes(attrs);
  assert(operands.size() == 1 && "Neg Op accept 1 operands");
  odsState.addTypes(operands[0].getType());
}

void mlir::spe::AddOp::build(::mlir::OpBuilder &odsBuilder,
                             ::mlir::OperationState &odsState,
                             ValueRange operands,
                             ArrayRef<NamedAttribute> attrs) {
  odsState.addOperands(operands);
  odsState.addAttributes(attrs);
  assert(operands.size() == 2 && "Add Op accept 2 operands");
  odsState.addTypes(operands[0].getType());
}

void mlir::spe::MulOp::build(::mlir::OpBuilder &odsBuilder,
                             ::mlir::OperationState &odsState,
                             ValueRange operands,
                             ArrayRef<NamedAttribute> attrs) {
  odsState.addOperands(operands);
  odsState.addAttributes(attrs);
  assert(operands.size() == 2 && "Mul Op accept 2 operands");
  odsState.addTypes(operands[0].getType());
}

mlir::Type mlir::spe::SpeDialect::parseType(
    ::mlir::DialectAsmParser &parser) const {
  parser.emitError(parser.getNameLoc(), "unknown type in SpeDialect");
  return mlir::Type();
}

void mlir::spe::SpeDialect::printType(Type type, DialectAsmPrinter &os) const {
  // If no custom types, nothing to print
  llvm_unreachable("SpeDialect has no types to print");
}

void mlir::spe::SpeDialect::registerTypes() {
  // If you plan to register types later, call addTypes(...)
}