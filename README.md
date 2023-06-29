# GEMM MMA

GEMM MMA 构建了一个初级的GEMM kernel， 它使用CUDA `mma.sync`指令来使用GPU tensor core单元，并对比了和cutlass算子的性能，本例主要为了介绍使用 `mma.sync` 构建一个完整的GEMM kernel，性能还有很大的优化空间。

ldmatrix 分支尝试使用`ldmatrix`指令来加载共享内存

## 总体思路

<img src="pic/gemm_vec.png" title="" alt="" width="600">

上图展示了GEMM MMA的计算流程，蓝色部分代表1个block要计算的部分，蓝色部分下的每个小方块代表每个warp的计算部分，右侧青色部分代表每个warp的计算部分，青色部分下的每个小方块代表tensor core支持的分块大小，在调用tensor core之前，加载一个绿色方块和红色方块进入共享内存，之后每个warp独立同步地调用`mma.sync` 来计算每个分块的结果，其中 $M'$ $N'$ $K'$ 代表tensor core单元支持计算的GEMM维度。

## ldmatrix

`ldmatrix.sync` 指令是 Warp-level matrix load instruction，它是 `mma.sync` 对应的load共享内存的指令

不过令人比较疑惑的是，在使用 `ldmatrix.sync` 后性能出现了小幅下降

## 结果

```
[        problem size] (8192,8192,8192)
[          cutlassMMA] Runtime: 12.580147(ms) Gflops: 87400.540405
[            MMA_tune] Runtime: 20.288921(ms) Gflops: 54192.709828
[       MMA_tune==ref] PASS
```
