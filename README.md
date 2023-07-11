# GEMM MMA

GEMM MMA 构建了一个初级的GEMM kernel， 它使用CUDA `mma.sync`指令来使用GPU tensor core单元，并对比了和cutlass算子的性能，本例主要为了介绍使用 `mma.sync` 构建一个完整的GEMM kernel，性能还有很大的优化空间。

epilogue 分支增加了对参数`alpha`和`beta`的支持

## 总体思路

<img src="pic/gemm_vec.png" title="" alt="" width="600">

上图展示了GEMM MMA的计算流程，蓝色部分代表1个block要计算的部分，蓝色部分下的每个小方块代表每个warp的计算部分，右侧青色部分代表每个warp的计算部分，青色部分下的每个小方块代表tensor core支持的分块大小，在调用tensor core之前，加载一个绿色方块和红色方块进入共享内存，之后每个warp独立同步地调用`mma.sync` 来计算每个分块的结果，其中 $M'$ $N'$ $K'$ 代表tensor core单元支持计算的GEMM维度。

## epilogue

The above code focuses only on the matrix multiply computation C = AB whose result is held in the registers of each thread within the threadblock. The mapping of logical elements in the output tile to each thread is chosen to maximize performance of the matrix multiply computation but does not result in efficient, coalesced loads and stores to global memory.

The epilogue is a separate phase in which threads exchange data through shared memory then cooperatively access global memory using efficient striped access patterns. It is also the phase in which linear scaling and other elementwise operations may be conveniently computed using the matrix product results as inputs.

CUTLASS defines several typical epilogue operations such as linear scaling and clamping, but other device-side function call operators may be used to perform custom operations.

之前的 kernel 只支持 alpha 和 beta 均设置成 1，为了支持不同的 alpha 和 beta 设置，epilogue 分支增加了一个函数 epilogue(完全类比 Cutlass 中的 epilogue概念)来完成 `D = alpha * (AB) + beta * C` 运算。

## 结果

可以发现使用 epilogue 的 kernel 速度会变慢一些 

```
[        problem size] (8192,8192,8192)
[        (alpha,beta)] (1.10,1.20)
[          Iterations] 20
[          cutlassMMA] Runtime: 12.385588(ms) Gflops: 88773.472450
[            MMA_tune] Runtime: 18.208921(ms) Gflops: 60383.127680
[       MMA_tune==ref] PASS
```