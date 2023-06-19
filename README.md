# GEMM MMA

GEMM MMA 构建了一个初级的GEMM kernel， 它使用CUDA `mma.sync`指令来使用GPU tensor core单元，并对比了和cutlass算子的性能，本例主要为了介绍使用 `mma.sync` 构建一个完整的GEMM kernel，性能还有很大的优化空间。

reg 分支介绍使用寄存器来优化性能

## 总体思路

<img src="pic/gemm_vec.png" title="" alt="" width="600">
 

上图展示了GEMM MMA的计算流程，蓝色部分代表1个block要计算的部分，蓝色部分下的每个小方块代表每个warp的计算部分，右侧青色部分代表每个warp的计算部分，青色部分下的每个小方块代表tensor core支持的分块大小，在调用tensor core之前，加载一个绿色方块和红色方块进入共享内存，之后每个warp独立同步地调用`mma.sync` 来计算每个分块的结果，其中 $M'$ $N'$ $K'$ 代表tensor core单元支持计算的GEMM维度。

## CUDA中的寄存器

寄存器的概念可能对于高级编程者来说是比较陌生的，因为在编程中一般并不会刻意地声明要使用寄存器来做什么操作，因为编译器会帮我们处理好这个问题，这就导致了在编写CUDA算子时往往会忽略掉寄存器的使用，可以通过ncu或编译时设置编译参数来查看kernel中每个线程使用了几个寄存器，比如在我们对比的cutlass的kernel中每个线程使用了230个寄存器，但是本例中baseline的kernel中每个线程只使用了32个寄存器，所以可以考虑将频繁使用的`tileC`(也就是图中的蓝色部分)从共享内存转移到寄存器中。

如何使用寄存器？其实很简单，在kernel中声明变量或数组就可以(不过如果一个线程使用太多寄存器会发生register spilling，可以在编译好程序后反汇编查看下有没有local memory)

在代码中添加了
```
ElementOutput C_fragment[64];
```
并修改好相关逻辑后，再次编译发现每个线程使用了156个线程

## 杂项
之前方法优化效果不明显的原因应该是kernel性能瓶颈在别的地方，可以从这个寄存器优化后的版本尝试如果不采用向量化，不解决bank冲突或访存不合并，或者不采用异步拷贝，kernel的性能变化是怎样的？

## 结果

原来没有使用寄存器才是kernel性能差的主要原因...

```
[        problem size] (8192,8192,8192)
[          cutlassMMA] Runtime: 16.149094(ms) Gflops: 68085.036418
[            MMA_base] Runtime: 297.333862(ms) Gflops: 3697.902483
[       MMA_base==ref] PASS
[            MMA_tune] Runtime: 45.636402(ms) Gflops: 24092.863952
[       MMA_tune==ref] PASS
```


