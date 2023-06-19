# GEMM MMA

GEMM MMA 构建了一个初级的GEMM kernel， 它使用CUDA `mma.sync`指令来使用GPU tensor core单元，并对比了和cutlass算子的性能，本例主要为了介绍使用 `mma.sync` 构建一个完整的GEMM kernel，性能还有很大的优化空间。

prefetch 分支介绍使用数据预取来优化性能

## 总体思路

<img src="pic/gemm_vec.png" title="" alt="" width="600">
 

上图展示了GEMM MMA的计算流程，蓝色部分代表1个block要计算的部分，蓝色部分下的每个小方块代表每个warp的计算部分，右侧青色部分代表每个warp的计算部分，青色部分下的每个小方块代表tensor core支持的分块大小，在调用tensor core之前，加载一个绿色方块和红色方块进入共享内存，之后每个warp独立同步地调用`mma.sync` 来计算每个分块的结果，其中 $M'$ $N'$ $K'$ 代表tensor core单元支持计算的GEMM维度。

## 数据预取

数据预取需要将缓冲区加倍，主要流程如下

假设计算`mma_{i}`依赖于数据`data_{i}`, `load data_{i}`代表开始加载数据`data_{i}`, 只有在`synchronize`后加载的数据才保证可见, 那么数据预取的伪代码如下

```
load data_{1}

for i=1:...
    synchronize
    mma_{i}
    load data_{i+1}
end
```
这样可以让数据加载(data_{i+1})和计算(mma_{i})尽可能重叠起来

## 结果

```
[        problem size] (8192,8192,8192)
[          cutlassMMA] Runtime: 15.947670(ms) Gflops: 68944.969952
[            MMA_base] Runtime: 45.381512(ms) Gflops: 24228.184273
[       MMA_base==ref] PASS
[            MMA_tune] Runtime: 40.519497(ms) Gflops: 27135.372140
[       MMA_tune==ref] PASS
```


