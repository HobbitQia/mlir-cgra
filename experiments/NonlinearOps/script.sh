mlir-opt --canonicalize -convert-tensor-to-linalg -linalg-init-tensor-to-alloc-tensor -eliminate-alloc-tensors   -linalg-bufferize -arith-bufferize   -tensor-bufferize -func-bufferize   -finalizing-bufferize -buffer-deallocation   --buffer-results-to-out-params   --canonicalize -cse 02-linalg.mlir > 03-finalized.mlir

soda-opt --convert-linalg-matmul-to-cgra --convert-nonlinear-to-cgra  03-finalized.mlir > 04-locating.mlir

soda-opt -outline-cgra-code -generate-cgra-hostcode 04-locating.mlir > 05-host.mlir