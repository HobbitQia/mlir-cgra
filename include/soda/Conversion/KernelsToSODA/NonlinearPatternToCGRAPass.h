//===- PatternToCGRAPass.h - Pass performing the pattern matching for CGRA C++ -*-=//
//===---------------------------------------------------------------------------===//
#ifndef MLIR_NONLINEAR_PATTERN_TO_CGRAPASS_H_
#define MLIR_NONLINEAR_PATTERN_TO_CGRAPASS_H_

#include "mlir/Support/LLVM.h"

#include <memory>

namespace mlir {
namespace func {
class FuncOp;
}
template <typename T>
class OperationPass;
class Pass;

std::unique_ptr<OperationPass<func::FuncOp>> createNonlinearPatternToCGRAPass();

} // namespace mlir

#endif // MLIR_NONLINEAR_PATTERN_TO_CGRAPASS_H_
