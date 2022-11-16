
# 02-linalg.mlir is generated by the bert.py located in model folder
mlir-opt --canonicalize -convert-tensor-to-linalg -linalg-init-tensor-to-alloc-tensor -eliminate-alloc-tensors   -linalg-bufferize -arith-bufferize   -tensor-bufferize -func-bufferize   -finalizing-bufferize -buffer-deallocation   --buffer-results-to-out-params   --canonicalize -cse 02-linalg.mlir > 03-finalized.mlir

soda-opt -lower-all-to-llvm 03-finalized.mlir > 04-llvm.mlir

mlir-translate --mlir-to-llvmir 04-llvm.mlir > 05-model.ll

llc -filetype=obj 05-model.ll

# also need to append llvm-project/build/lib/ on LD_LIBRARY_PATH:
# export LD_LIBRARY_PATH=../../../../llvm-project/build/lib/
clang++-12 main.cpp 05-model.o ../../../../llvm-project/build/lib/libmlir_c_runner_utils.so -I../../../sim/ ../../../sim/*.cpp -o simulate

./simulate 4 false true
