# GEMM MMA

GEMM MMA 构建了一个初级的GEMM kernel， 它使用CUDA `mma.sync`指令来使用GPU tensor core单元，并对比了和cutlass算子的性能，本例主要为了介绍使用 `mma.sync` 构建一个完整的GEMM kernel，性能还有很大的优化空间。

## Optimize GEMM step by step

一步步优化GEMM系列，每次引入一个优化概念并对比性能变化，代码在每个分支的`gemm.cu`

baseline性能: 3.44% (相比cutlass)

### [1. 使用向量化(vector)](https://github.com/gty111/GEMM_MMA/tree/vector)

vector分支主要介绍向量化load/store，

优化后性能: 4.74%

### [2. 避免bank冲突并且合并访存(bfco)](https://github.com/gty111/GEMM_MMA/tree/bfco)

bfco分支主要介绍如何通过解决shared memory bank conflict 和 memory coalesce (访存合并) 来优化性能

优化后性能: 5.00%

### [3. 使用异步拷贝(ldgsts)](https://github.com/gty111/GEMM_MMA/tree/ldgsts)

ldgsts 分支主要来介绍使用Ampere引入的异步拷贝来优化性能

优化后性能: 5.36%

### [4. 使用寄存器(reg)](https://github.com/gty111/GEMM_MMA/tree/reg)

reg 分支介绍使用寄存器来优化性能

优化后性能: 35.39%

### [5. 使用数据预取(prefetch)](https://github.com/gty111/GEMM_MMA/tree/prefetch)

prefetch 分支介绍使用数据预取来优化性能

优化后性能：39.36%

### [6. 关于PTXAS有趣的发现(ptxas)](https://github.com/gty111/GEMM_MMA/tree/ptxas)

ptxas 分支分享一个调优过程中发现的关于ptxas(ptx汇编器)有意思的东西

### [7. 优化数据预取(prefetchx)](https://github.com/gty111/GEMM_MMA/tree/prefetchx)

prefetchx 分支和之前的prefetch分支类似，区别是增加了预取数据大小并利用了同步指令`cp.async.waitgroup N`

优化后性能：46.89%

### [8. 调整线程块和warp计算的矩阵大小(shape)](https://github.com/gty111/GEMM_MMA/tree/shape)

shape 分支调整了每个block和warp计算的矩阵C的大小

优化后性能：62.39%

### [9. 调整线程块分配到的计算位置(swizzle)](https://github.com/gty111/GEMM_MMA/tree/swizzle)

swizzle 分支调整每个thread block分配到的计算位置来优化性能

优化后性能: 68.43%

### [10. 尝试使用ldmatrix指令(ldmatrix)](https://github.com/gty111/GEMM_MMA/tree/ldmatrix)

ldmatrix 分支尝试使用`ldmatrix`指令来加载共享内存