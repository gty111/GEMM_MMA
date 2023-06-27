# GEMM MMA

GEMM MMA 构建了一个初级的GEMM kernel， 它使用CUDA `mma.sync`指令来使用GPU tensor core单元，并对比了和cutlass算子的性能，本例主要为了介绍使用 `mma.sync` 构建一个完整的GEMM kernel，性能还有很大的优化空间。

shape 分支调整了每个block和warp计算的矩阵C的大小

## 总体思路

<img src="pic/gemm_vec.png" title="" alt="" width="600">

上图展示了GEMM MMA的计算流程，蓝色部分代表1个block要计算的部分，蓝色部分下的每个小方块代表每个warp的计算部分，右侧青色部分代表每个warp的计算部分，青色部分下的每个小方块代表tensor core支持的分块大小，在调用tensor core之前，加载一个绿色方块和红色方块进入共享内存，之后每个warp独立同步地调用`mma.sync` 来计算每个分块的结果，其中 $M'$ $N'$ $K'$ 代表tensor core单元支持计算的GEMM维度。


## shape

cutlass 中 `ShapeMMAThreadBlock` 代表每个线程块计算的矩阵C的大小，而 `ShapeMMAWarp` 代表每个warp计算的矩阵C的大小

之前手写的 MMA kernel 每个线程块计算 128x64，每个warp计算 64x32，调整后每个线程块计算 128x128，每个warp计算 64x64，这样可以增加数据复用

PS: 实测如果在kernel中申请长度为128的数组，编译器会将其分配到local memory， 所以为了避免这样的情况发生，需要将长度为128的数组分成两个长度为64的数组

## 结果

> PS: 这次测试结果没有对比之前的kernel

```
[        problem size] (8192,8192,8192)
[          cutlassMMA] Runtime: 12.744192(ms) Gflops: 86275.506296
[            MMA_tune] Runtime: 20.427467(ms) Gflops: 53825.156547
[       MMA_tune==ref] PASS
```
