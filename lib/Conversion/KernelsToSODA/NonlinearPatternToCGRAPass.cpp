//===- OperationToSODAPass.cpp - Convert named ops into SODA operations ---- =//
//
// This pass converts different operations that match the selected named into
// soda.launch + the same operation. Marking the region to be outlined.
//
//===----------------------------------------------------------------------===//

#include "soda/Conversion/KernelsToSODA/NonlinearPatternToCGRAPass.h"
#include "../PassDetail.h"
#include "soda/Conversion/KernelsToSODA/NonlinearPatternToCGRA.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"
#include "soda/Dialect/SODA/SODADialect.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/CommandLine.h"

#include <list>
#include <vector>

using namespace std;
using namespace mlir;

namespace {

struct NonlinearOperationMapper : public ConvertNonlinearPatternToCGRABase<NonlinearOperationMapper> {
  NonlinearOperationMapper() = default;

  void runOnInnerOp(scf::ForOp& forOp) {

    for (Operation &innerOp : llvm::make_early_inc_range(forOp.getBody()->getOperations())) {
      if (auto op = dyn_cast<scf::ForOp>(&innerOp)) {
        runOnInnerOp(op);
      } else {
        // if (innerOp.getName().getStringRef() == targetPatterns) {
        if (auto genericOp = dyn_cast<linalg::GenericOp>(&innerOp)) {
          if (failed(convertNonlinearPatternToCGRALaunch(&innerOp)))
            signalPassFailure();
	}
      }
    }
  }

  void runOnOperation() override {
    auto funcOp = getOperation();
    
    // deal with element-wise cases
    for (Operation &op : llvm::make_early_inc_range(funcOp.getOps())) {
      // if (op.getName().getStringRef() == targetPatterns) {
      if (auto genericOp = dyn_cast<linalg::GenericOp>(&op)) {
        if (failed(convertNonlinearPatternToCGRALaunch(&op)))
          signalPassFailure();
      } else if (auto forOp = dyn_cast<scf::ForOp>(&op)) {
        runOnInnerOp(forOp);
      }
    }
    // deal with mu
    vector<vector<string>> Patterns;
    vector<string> softmax = {"index-index_cast-max-cmp-select", "sub", "exp", "add", "div"};
    vector<string> layernorm = {"add", "div", "", "sub", "mul", "add", "div", "trunc-add", "rsqrt", "", "mul", "mul", "add"};
    vector<string> swiglu = {"neg-exp-add-div", "mul", "mul"};
    Patterns.push_back(softmax); Patterns.push_back(layernorm); Patterns.push_back(swiglu);
    vector<string> patternNames = {"softmax", "layernorm", "swiglu"};
    for (auto matchedPatterns : Patterns) {
      int size = matchedPatterns.size();
      for (Operation &now_op : llvm::make_early_inc_range(funcOp.getOps())) {
        bool flag = true;
        int i = 0;
        Operation *op = &now_op;
        Operation *start_op = nullptr;
        Operation *nextOp = nullptr;
        Operation *st = nullptr;
        Operation *ed = nullptr;
        list<Operation *> tmp_locs;
        if (!(dyn_cast<linalg::GenericOp>(op))) continue;
        while (true) {
            bool bj = false;
            // Operation &refOp = *op;
            if (auto genericOp = dyn_cast<linalg::GenericOp>(op)) {
              bj = tryMatchedPattern(op, matchedPatterns[i]);
              // if (bj) break;
              // printf("%s", refOp.getName().getStringRef());
              i++;
              flag &= bj;
              if (i == 1) st = op;
              else if (i == size) ed = op;
              if (i == size) start_op = op;
              else tmp_locs.push_back(op);
              // regions.push_back(genericOp.getRegion());
            }
            nextOp = op->getNextNode();
            if (flag && i == size) {
              printf("NB\n");
              bool bj = false;
              Operation *newOp = Merge(start_op, st, ed, patternNames.front());
              for (auto tmp_op : tmp_locs) {
                  printf("delete\n");
                  auto a = newOp->getResults();
                  printf("newOp %s %s\n", tmp_op->getResults(), a);
                  tmp_op->replaceAllUsesWith(newOp);
                  // break;
                  tmp_op->erase();
                
              }
              break;
            }
            if (!flag || i == size || nextOp == nullptr) {
              break;
            }
            op = nextOp;
          }
        }
        patternNames.erase(patternNames.begin());
      }
    }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> mlir::createNonlinearPatternToCGRAPass() {
  return std::make_unique<NonlinearOperationMapper>();
}