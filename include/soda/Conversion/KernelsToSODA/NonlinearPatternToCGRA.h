//===- OperationToCGRA.h - Convert Named Operations to CGRA kernels C++ -*-===//
//===----------------------------------------------------------------------===//
#ifndef MLIR_NONLINEAR_PATTERN_TO_CGRA_H_
#define MLIR_NONLINEAR_PATTERN_TO_CGRA_H_

#include "mlir/Support/LLVM.h"
#include <string>

namespace mlir {
struct LogicalResult;

class Operation;

/// Convert Operations that match opName into soda.
LogicalResult convertNonlinearPatternToCGRALaunch(Operation *op);

bool tryMatchedPattern(Operation *op, std::string pattern);
Operation* Merge(Operation *op, Operation *st, Operation *ed, std::string pattern);

} // namespace mlir

#endif // MLIR_NONLINEAR_PATTERN_TO_CGRA_H_
