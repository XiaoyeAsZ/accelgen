/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: SpeOps.td                                                            *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::spe::SpeDialect)
namespace mlir {
namespace spe {

SpeDialect::SpeDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<SpeDialect>())
    
     {
  
  initialize();
}

SpeDialect::~SpeDialect() = default;

} // namespace spe
} // namespace mlir
