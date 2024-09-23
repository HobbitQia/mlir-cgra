//===- OperationToSODA.cpp - Convert operations matching string to a SODA -===//
//===----------------------------------------------------------------------===//
//
// This implements a straightforward conversion of operation that matches anchor
// string to SODA launch operations.
//
//===----------------------------------------------------------------------===//

#include "soda/Conversion/KernelsToSODA/PatternToCGRA.h"
#include "soda/Dialect/SODA/SODADialect.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"

#include <iostream>
#include <string>
#include <vector>

#define DEBUG_TYPE "pattern-to-cgra"

using namespace std;
using namespace mlir;

namespace {
// Helper structure that holds common state of the loop to CGRA kernel
// conversion.
struct PatternToCGRAConverter {
  void createLaunch(Operation *op, ArrayRef<string> patterns);
  string getMatchedPattern(std::vector<string> &arithOptNames, string pattern);
};

// struct NonlinearPatternToCGRAConverter {
//   void createLaunch(Operation *op);
// };

} // namespace

//===----------------------------------------------------------------------===//
// PatternToCGRA
//===----------------------------------------------------------------------===//

string PatternToCGRAConverter::getMatchedPattern(std::vector<string> &arithOptNames, string pattern) {

  // One pattern (i.e., operation chain) includes multiple ops that are connected
  // by `-` as delimiter.
  std::vector<string> patternOpts;
  string separator = "-";
  auto start = 0;
  auto end = pattern.find(separator);
  while (end != string::npos) {
    patternOpts.push_back(pattern.substr(start, end - start));
    start = end + separator.length();
    end = pattern.find(separator, start);
  }
  patternOpts.push_back(pattern.substr(start, end - start));

  if (arithOptNames.size() != patternOpts.size())
    return "";

  string matchedPattern = "";
  for (auto i=0; i<(int)arithOptNames.size(); ++i) {
    if (arithOptNames[i].find(patternOpts[i]) == string::npos) {
      return "";
    }
    if (i != 0)
      matchedPattern += "_";
    matchedPattern += patternOpts[i];
  }
  return matchedPattern;
}

/*
void NonlinearPatternToCGRAConverter::createLaunch(Operation *op) {

  PatternToCGRAConverter converter;
  OpBuilder builder(op);
  auto genericOp = dyn_cast<linalg::GenericOp>(op);

  // printf
  // std::cout << "genericOp: " <<  "\n";

  // potentially create a launch op and move target op into the region
  Location loc = genericOp->getLoc();

  std::vector<string> arithOptNames;
  for (Operation &arithOp : llvm::make_early_inc_range(genericOp.getRegion().front().getOperations())) {
    string arithOptName = string(arithOp.getName().getStringRef());
    // ignore linalg.yield as it is not a computation
    if (arithOptName != "linalg.yield")
      arithOptNames.push_back(arithOptName);
  }
  string patterns[5] = {"div-erf-add-mul-mul"};
  string ops[5] = {"GELU"};
  for (auto i: llvm::seq(0, 1)) {
    std::cout << "pattern: " << patterns[i] << "\n";
    string pattern = patterns[i];
    string matchedPattern = converter.getMatchedPattern(arithOptNames, pattern);
    if (matchedPattern != "") {
      auto launchOp = builder.create<soda::LaunchOp>(loc);
      builder.setInsertionPointToEnd(&launchOp.body().front());
      builder.create<soda::TerminatorOp>(loc);
      builder.setInsertionPointToStart(&launchOp.body().front());

      auto *newOp = Operation::create(
        genericOp->getLoc(), genericOp->getName(),
        genericOp->getResultTypes(), genericOp->getOperands(),
        genericOp->getAttrDictionary(), genericOp->getSuccessors(),
        genericOp->getRegions());

      launchOp->setAttr("pattern",
                  StringAttr::get(builder.getContext(), ops[i]));

      // Insert the clone into the soda launch.
      auto results = newOp->getResults();
      builder.insert(newOp);
      genericOp->replaceAllUsesWith(results);
      genericOp->erase();
      // break;

    }
  }
}*/

// Add a SODA launch operation around the generic op if the inner ops match
// the pattern.
void PatternToCGRAConverter::createLaunch(Operation *op, ArrayRef<string> patterns) {

  OpBuilder builder(op);
  auto genericOp = dyn_cast<linalg::GenericOp>(op);

  // potentially create a launch op and move target op into the region
  Location loc = genericOp->getLoc();

  std::vector<string> arithOptNames;
  for (Operation &arithOp : llvm::make_early_inc_range(genericOp.getRegion().front().getOperations())) {
    string arithOptName = string(arithOp.getName().getStringRef());
    // ignore linalg.yield as it is not a computation
    if (arithOptName != "linalg.yield")
      arithOptNames.push_back(arithOptName);
  }

  for (auto pattern: patterns) {
    string matchedPattern = getMatchedPattern(arithOptNames, pattern);
    if (matchedPattern != "") {
      auto launchOp = builder.create<soda::LaunchOp>(loc);
      builder.setInsertionPointToEnd(&launchOp.body().front());
      builder.create<soda::TerminatorOp>(loc);
      builder.setInsertionPointToStart(&launchOp.body().front());

      auto *newOp = Operation::create(
        genericOp->getLoc(), genericOp->getName(),
        genericOp->getResultTypes(), genericOp->getOperands(),
        genericOp->getAttrDictionary(), genericOp->getSuccessors(),
        genericOp->getRegions());

      launchOp->setAttr("pattern",
                  StringAttr::get(builder.getContext(), "fusion_" + matchedPattern));

      // Insert the clone into the soda launch.
      auto results = newOp->getResults();
      builder.insert(newOp);
      genericOp->replaceAllUsesWith(results);
      genericOp->erase();
      break;

    }
  }
}

static LogicalResult convertPatternToCGRALaunch(Operation *op, ArrayRef<string> patterns) {
  PatternToCGRAConverter converter;
  converter.createLaunch(op, patterns);
  return success();
}

// static LogicalResult convertNonlinearPatternToCGRALaunch(Operation *op) {
//   NonlinearPatternToCGRAConverter converter;
//   converter.createLaunch(op);
//   return success();
// }

LogicalResult mlir::convertPatternToCGRALaunch(Operation *op, ArrayRef<string> patterns) {
  // if (patterns.empty())
  //   return ::convertNonlinearPatternToCGRALaunch(op);
  return ::convertPatternToCGRALaunch(op, patterns);
}

// bool mlir::tryMatchedPattern(Operation *op, std::string pattern) {
  
//   PatternToCGRAConverter converter;
//   std::cout << "op: " << pattern << "\n";
  
//   auto genericOp = dyn_cast<linalg::GenericOp>(op);

//   // potentially create a launch op and move target op into the region
//   Location loc = genericOp->getLoc();

//   std::vector<string> arithOptNames;
//   for (Operation &arithOp : llvm::make_early_inc_range(genericOp.getRegion().front().getOperations())) {
//     string arithOptName = string(arithOp.getName().getStringRef());
//     // ignore linalg.yield as it is not a computation
//     std::cout << "arithOpName: " << arithOptName << "\n";
//     if (arithOptName != "linalg.yield")
//       arithOptNames.push_back(arithOptName);
//   }
//   if (arithOptNames.size() == 0 && pattern == "") return true;

//   {
//     string matchedPattern = converter.getMatchedPattern(arithOptNames, pattern);
//     std::cout << "Matched pattern: " << matchedPattern << "\n";
//     if (matchedPattern != "") {
//       return true;
//     }
//   }
//   return false;
// }

// Operation* mlir::Merge(Operation *op, Operation *st, Operation *ed, std::string pattern) {

//   OpBuilder builder(op);
//   auto genericOp = dyn_cast<linalg::GenericOp>(op);
//   auto st_generic = dyn_cast<linalg::GenericOp>(st);
//   auto ed_generic = dyn_cast<linalg::GenericOp>(ed);

//   // potentially create a launch op and move target op into the region
//   Location loc = genericOp->getLoc();

//   auto launchOp = builder.create<soda::LaunchOp>(loc);
//   builder.setInsertionPointToEnd(&launchOp.body().front());
//   builder.create<soda::TerminatorOp>(loc);
//   builder.setInsertionPointToStart(&launchOp.body().front());
//   std::cout << "create launchOp\n";
//   std::cout << st_generic->getOperands().size() << " " << ed_generic->getOperands().size() << " ";
//   std::cout << genericOp->getOperands().size() << "\n";
//   std::cout << st_generic->getResults().size() << " " << ed_generic->getResults().size() << " ";
//   std::cout << genericOp->getResults().size() << "\n";
//   auto *newOp = Operation::create(
//     genericOp->getLoc(), genericOp->getName(),
//     // op(1) is unused.
//     genericOp->getResultTypes(), {st_generic->getOperand(0), st_generic->getOperand(0), ed_generic->getOperand(2)},
//     genericOp->getAttrDictionary(), genericOp->getSuccessors(),
//     genericOp->getRegions());
//   // for (auto region: regions) {
//   //     newOp->getRegion().push_back(std::move(region));
//   // }
//   std::cout << "create succesfully\n";
//   launchOp->setAttr("pattern",
//               StringAttr::get(builder.getContext(), pattern));

//   // Insert the clone into the soda launch.
//   auto results = newOp->getResults();
//   builder.insert(newOp);
//   genericOp->replaceAllUsesWith(results);
//   genericOp->erase();

//   auto save_op = st;

//   while (true) {
//     auto nextOp = st->getNextNode();
//     if (st == ed) {
//       break;
//     }
//     if ((st->getName().getStringRef() == "memref.dealloc") && 
//         st->getOperand(0) == st_generic->getOperand(0))
//         {
//           st->erase();
//           break;
//         }
//     st = nextOp;
//   }


//   return newOp;
// }