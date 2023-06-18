# GEMM MMA

GEMM MMA 构建了一个初级的GEMM kernel， 它使用CUDA `mma.sync`指令来使用GPU tensor core单元，并对比了和cutlass算子的性能，本例主要为了介绍使用 `mma.sync` 构建一个完整的GEMM kernel，性能还有很大的优化空间。

## [切换到vector分支](https://github.com/gty111/GEMM_MMA/tree/vector)

vector分支主要介绍向量化load/store

## 总体思路

<img src="pic/gemm_mma.png" title="" alt="" width="491">
 

上图展示了GEMM MMA的计算流程，蓝色部分代表1个block要计算的部分，蓝色部分下的每个小方块代表每个warp的计算部分，在调用tensor core之前，加载一个绿色方块和红色方块进入共享内存，之后每个warp独立同步地调用`mma.sync` 来计算每个分块的结果，其中 $M'$ $N'$ $K'$ 代表tensor core单元支持计算的GEMM维度。



其中每个block内warp的排布有多种选择(图中为3x2)：

{1,2,3,4}x{1,2,3,4}，即 1x1,1x2,1x3,1x4,2x1,2x2, 2x3, 2x4 ... 4x3, 4x4



## 结果

其中 MMA_i_j 代表block内warp的排布为 ixj

```
[        problem size] (5120,4096,4096)
[          cutlassMMA] Runtime: 2.405991(ms) Gflops: 71404.556526
[             MMA_1_1] Runtime: 82.386307(ms) Gflops: 2085.282113
[ cutlassMMA==MMA_1_1] PASS
[             MMA_1_2] Runtime: 79.756462(ms) Gflops: 2154.041031
[ cutlassMMA==MMA_1_2] PASS
[             MMA_1_3] Runtime: 67.411049(ms) Gflops: 2548.524236
[ cutlassMMA==MMA_1_3] PASS
[             MMA_1_4] Runtime: 66.942772(ms) Gflops: 2566.351630
[ cutlassMMA==MMA_1_4] PASS
[             MMA_2_1] Runtime: 41.768246(ms) Gflops: 4113.141191
[ cutlassMMA==MMA_2_1] PASS
[             MMA_2_2] Runtime: 34.134132(ms) Gflops: 5033.046978
[ cutlassMMA==MMA_2_2] PASS
[             MMA_2_3] Runtime: 33.721062(ms) Gflops: 5094.699963
[ cutlassMMA==MMA_2_3] PASS
[             MMA_2_4] Runtime: 31.809515(ms) Gflops: 5400.858575
[ cutlassMMA==MMA_2_4] PASS
[             MMA_3_1] Runtime: 34.388866(ms) Gflops: 4995.764900
[ cutlassMMA==MMA_3_1] PASS
[             MMA_3_2] Runtime: 27.578543(ms) Gflops: 6229.433283
[ cutlassMMA==MMA_3_2] PASS
[             MMA_3_3] Runtime: 26.916382(ms) Gflops: 6382.681480
[ cutlassMMA==MMA_3_3] PASS
[             MMA_3_4] Runtime: 25.842243(ms) Gflops: 6647.979068
[ cutlassMMA==MMA_3_4] PASS
[             MMA_4_1] Runtime: 39.947754(ms) Gflops: 4300.584515
[ cutlassMMA==MMA_4_1] PASS
[             MMA_4_2] Runtime: 31.890951(ms) Gflops: 5387.067040
[ cutlassMMA==MMA_4_2] PASS
[             MMA_4_3] Runtime: 32.361828(ms) Gflops: 5308.683200
[ cutlassMMA==MMA_4_3] PASS
[             MMA_4_4] Runtime: 31.914679(ms) Gflops: 5383.061949
[ cutlassMMA==MMA_4_4] PASS
```


