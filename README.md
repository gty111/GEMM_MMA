# GEMM MMA

GEMM MMA 构建了一个初级的GEMM kernel， 它使用CUDA `mma.sync`指令来使用GPU tensor core单元，并对比了和cutlass算子的性能，本例主要为了介绍使用 `mma.sync` 构建一个完整的GEMM kernel，性能还有很大的优化空间。

swizzle 分支调整每个thread block分配到的计算位置来优化性能

## 总体思路

<img src="pic/gemm_vec.png" title="" alt="" width="600">

上图展示了GEMM MMA的计算流程，蓝色部分代表1个block要计算的部分，蓝色部分下的每个小方块代表每个warp的计算部分，右侧青色部分代表每个warp的计算部分，青色部分下的每个小方块代表tensor core支持的分块大小，在调用tensor core之前，加载一个绿色方块和红色方块进入共享内存，之后每个warp独立同步地调用`mma.sync` 来计算每个分块的结果，其中 $M'$ $N'$ $K'$ 代表tensor core单元支持计算的GEMM维度。


## swizzle

本次优化思路来自于 cutlass 中 `ThreadBlockSwizzle`，一开始接触可能比较难以理解这个概念，这个swizzle最核心的就是对于 `blockIdx` 进行如下变换

```
blockIdx.x ==> block_idx_x >> log_tile
blockIdx.y ==> (block_idx_y << log_tile) + ((block_idx_x) & ((1 << (log_tile)) - 1))
blockIdx.z ==> blockIdx.z
```

看了上面的变换公式，你可能还是一头雾水，其实我们来用一张图来简单说明(log_tile=2), 假如我们启动了一个kernel，它的gridDim为(16,1,1)(即下图中左边的分布)，那么经过 `swizzle` 变换后得到下图中右边的分布，所以我们可以发现 swizzle 后的线程块在2D分布上满足局部性原理，那这样有什么好处呢，好处就是可以尽量提升L2缓存的命中率或L2缓存中的数据复用。

```
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15  |  0  4  8  12
                                       |  1  5  9  13 
                                       |  2  6 10  14
                                       |  3  7 11  15
```



## 结果

```
[        problem size] (8192,8192,8192)
[          cutlassMMA] Runtime: 12.671488(ms) Gflops: 86770.523274
[            MMA_tune] Runtime: 18.476339(ms) Gflops: 59509.170487
[       MMA_tune==ref] PASS
```
