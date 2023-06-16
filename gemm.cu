#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/kernel/default_gemm.h"
#include "cutlass/gemm/kernel/gemm.h"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_elementwise.h"
#include "cutlass/util/tensor_view_io.h"

#include "helper.h"

#include <iostream>

// The code section below describes datatype for input, output matrices and computation between
// elements in input matrices.
using ElementAccumulator = float;                   // <- data type of accumulator
using ElementComputeEpilogue = ElementAccumulator;  // <- data type of epilogue operations
using ElementInputA = cutlass::tfloat32_t;          // <- data type of elements in input matrix A
using ElementInputB = cutlass::tfloat32_t;          // <- data type of elements in input matrix B
using ElementOutput = float;                        // <- data type of elements in output matrix D

// The code section below describes matrix layout of input and output matrices. Column Major for
// Matrix A, Row Major for Matrix B and Row Major for Matrix C
using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::RowMajor;

// This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
using MMAOp = cutlass::arch::OpClassTensorOp;

// This code section describes CUDA SM architecture number
using SmArch = cutlass::arch::Sm80;

// This code section describes the tile size a thread block will compute
using ShapeMMAThreadBlock =
    cutlass::gemm::GemmShape<128, 128, 16>;  // <- threadblock tile M = 128, N = 128, K = 16
// This code section describes tile size a warp will compute
using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 16>;  // <- warp tile M = 64, N = 64, K = 16
// This code section describes the size of MMA op
using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>;  // <- MMA Op tile M = 16, N = 8, K = 8

// This code section describes how threadblocks are scheduled on GPU
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;  // <- ??

// This code section describes the epilogue part of the kernel
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,                                     // <- data type of output matrix
    128 / cutlass::sizeof_bits<ElementOutput>::value,  // <- the number of elements per vectorized
                                                       // memory access. For a byte, it's 16
                                                       // elements. This becomes the vector width of
                                                       // math instructions in the epilogue too
    ElementAccumulator,                                // <- data type of accumulator
    ElementComputeEpilogue>;  // <- data type for alpha/beta in linear combination function

// Number of pipelines you want to use
constexpr int NumStages = 4;

using Gemm = cutlass::gemm::device::Gemm<ElementInputA,
                                         LayoutInputA,
                                         ElementInputB,
                                         LayoutInputB,
                                         ElementOutput,
                                         LayoutOutput,
                                         ElementAccumulator,
                                         MMAOp,
                                         SmArch,
                                         ShapeMMAThreadBlock,
                                         ShapeMMAWarp,
                                         ShapeMMAOp,
                                         EpilogueOp,
                                         SwizzleThreadBlock,
                                         NumStages>;


const int blockdim = 128;
const int M = ShapeMMAOp::kM;
const int N = ShapeMMAOp::kN;
const int K = ShapeMMAOp::kK;

struct MMAarguments{
    cutlass::gemm::GemmCoord problem_size;
    ElementInputA *A;
    ElementInputB *B;
    ElementAccumulator *C;
    ElementOutput *D;
};

__device__ void loadtileA(MMAarguments &arg,ElementInputA *tileA,int idx){
    const int iter = 2*M*K / blockdim;
    CUTLASS_PRAGMA_UNROLL
    for(int i=0;i<iter;i++){
        int tileIdx = threadIdx.x*iter + i;
        int rowA = blockIdx.x*2*M + tileIdx/K ;
        int colA = idx*K + tileIdx%K ;
        tileA[tileIdx] = rowA<arg.problem_size.m() && colA<arg.problem_size.k() ? arg.A[rowA*arg.problem_size.k()+colA] : ElementInputA(0);
    }
}

__device__ void loadtileB(MMAarguments &arg,ElementInputB *tileB,int idx){
    const int iter = 2*N*K/blockdim;
    CUTLASS_PRAGMA_UNROLL
    for(int i=0;i<iter;i++){
        int tileIdx = threadIdx.x*iter + i;
        int rowB = idx*K + tileIdx%K ;
        int colB = blockIdx.y*2*N + tileIdx/K ;
        tileB[tileIdx] = rowB<arg.problem_size.k() && colB<arg.problem_size.n() ? arg.B[colB*arg.problem_size.k()+rowB] : ElementInputB(0);
    }
}

__device__ void loadtileC(MMAarguments &arg,ElementAccumulator *tileC){
    const int iter = 2*M*2*N/blockdim;
    CUTLASS_PRAGMA_UNROLL
    for(int i=0;i<iter;i++){
        int tileIdx = threadIdx.x*iter + i;
        int rowC = blockIdx.x*2*M + tileIdx/(2*N);
        int colC = blockIdx.y*2*N + tileIdx%(2*N);
        tileC[tileIdx] = rowC<arg.problem_size.m() && colC<arg.problem_size.n() ? arg.C[rowC*arg.problem_size.n()+colC] : ElementAccumulator(0);
    }
}

__device__ void mvtile(MMAarguments &arg,ElementAccumulator *dst,ElementOutput *src,int size){
    const int iter = size/blockdim;
    CUTLASS_PRAGMA_UNROLL
    for(int i=0;i<iter;i++){
        int tileIdx = threadIdx.x*iter + i;
        dst[tileIdx] = src[tileIdx]; 
    }
}

__device__ void mmatile(MMAarguments &arg,ElementInputA *A,ElementInputB *B,ElementAccumulator *C,ElementOutput *D){
    int warpidx = threadIdx.x / 32;
    int rowwarp = warpidx/2;
    int colwarp = warpidx%2;
    int laneidx = threadIdx.x % 32;

    int a[4],b[2],cd[4];

    cd[0] = (rowwarp*M+laneidx/4)*2*N+colwarp*N+laneidx%4*2;
    cd[1] = cd[0] + 1;
    cd[2] = cd[0] + 8*2*N;
    cd[3] = cd[2] + 1;

    a[0] = (rowwarp*M+laneidx/4)*K+laneidx%4;
    a[1] = a[0] + 8*K;
    a[2] = a[0] + 4;
    a[3] = a[1] + 4;

    b[0] = (colwarp*N+laneidx/4)*K+laneidx%4;
    b[1] = b[0] + 4;

    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : 
        "=f"(D[cd[0]]),  // D[0]
        "=f"(D[cd[1]]),  // D[1]
        "=f"(D[cd[2]]),  // D[2]
        "=f"(D[cd[3]])   // D[3]
        : 
        "r"(*reinterpret_cast<uint32_t const *>(&A[a[0]])),   // A[0]
        "r"(*reinterpret_cast<uint32_t const *>(&A[a[1]])),   // A[1]
        "r"(*reinterpret_cast<uint32_t const *>(&A[a[2]])),   // A[2]
        "r"(*reinterpret_cast<uint32_t const *>(&A[a[3]])),   // A[3]
        "r"(*reinterpret_cast<uint32_t const *>(&B[b[0]])),   // B[0]
        "r"(*reinterpret_cast<uint32_t const *>(&B[b[1]])),   // B[1]
        "f"(C[cd[0]]),   // C[0]
        "f"(C[cd[1]]),   // C[1]
        "f"(C[cd[2]]),   // C[2]
        "f"(C[cd[3]])    // C[3]
    );
}

__device__ void inittileD(ElementOutput *outD){
    const int iter = 2*M*2*N / blockdim;
    CUTLASS_PRAGMA_UNROLL
    for(int i=0;i<iter;i++){
        outD[i+threadIdx.x*iter] = 0;
    }
}

__device__ void storetileD(MMAarguments &arg,ElementOutput *outD){
    const int iter = 2*M*2*N/blockdim;
    CUTLASS_PRAGMA_UNROLL
    for(int i=0;i<iter;i++){
        int tileIdx = threadIdx.x*iter + i;
        int rowD = blockIdx.x*2*M + tileIdx/(2*N);
        int colD = blockIdx.y*2*N + tileIdx%(2*N);
        if(rowD<arg.problem_size.m() && colD<arg.problem_size.n()) {
            arg.D[rowD*arg.problem_size.n()+colD] = outD[tileIdx];
        }
    }
}

template<typename T>
__device__ void printtile(T *arr,int row,int col,bool rowMajor){
    if(threadIdx.x!=0||blockIdx.x!=0)return;
    for(int i=0;i<row;i++){
        for(int j=0;j<col;j++){
            if(rowMajor)
                printf("%d ",(int)arr[i*col+j]);
            else
                printf("%d ",(int)arr[j*row+i]);
        }
        printf("\n");
    }
}

__global__ void GEMM_MMA(MMAarguments arg){
    __shared__ ElementInputA tileA[M*2*K]; 
    __shared__ ElementInputB tileB[K*N*2]; 
    __shared__ ElementOutput tileC[2*M*2*N];
    __shared__ ElementOutput tileD[2*M*2*N];

    const int iters = (arg.problem_size.k() + K - 1) / K;

    // inittileD(tileD);
    loadtileC(arg,tileC);
    
    for(int idx=0;idx<iters;idx++){
        loadtileA(arg,tileA,idx);
        loadtileB(arg,tileB,idx);
        __syncthreads();
        mmatile(arg,tileA,tileB,tileC,tileD);
        __syncthreads();
        mvtile(arg,tileC,tileD,2*M*2*N);
    }
    __syncthreads();
    storetileD(arg,tileD);
}

void launch_GEMM_MMA(MMAarguments arg){
    dim3 grid,block;
    grid.x = (arg.problem_size.m()+M*2-1)/(M*2);
    grid.y = (arg.problem_size.n()+N*2-1)/(N*2);
    grid.z = 1;

    block.x = blockdim;
    block.y = 1;
    block.z = 1;

    GEMM_MMA<<<grid,block>>>(arg);
}

int main(int argc,char **argv){
    // Create a tuple of problem size for matrix multiplication
    cutlass::gemm::GemmCoord problem_size = {16,8,8};

    if(argc>=2)problem_size.m()=atoi(argv[1]);
    if(argc>=3)problem_size.n()=atoi(argv[2]);
    if(argc>=4)problem_size.k()=atoi(argv[3]);

    // Initialize tensors using CUTLASS helper functions
    cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(
      problem_size.mk());  // <- Create matrix A with dimensions M x K
    cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_b(
      problem_size.kn());  // <- Create matrix B with dimensions K x N
    cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_c(
      problem_size.mn());  // <- Create matrix C with dimensions M x N
    cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_d(
      problem_size.mn());  // <- Create matrix D with dimensions M x N used to store output from
                           // CUTLASS kernel
    cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_mma_d(
      problem_size.mn());  // <- Create matrix D with dimensions M x N used to store output from
                           // reference kernel
    
    cutlass::reference::device::TensorFillRandomUniform(
        tensor_a.device_view(),
        1,
        ElementInputA(4.f),
        ElementInputA(-4.f),
        0
    );

    cutlass::reference::device::TensorFillRandomUniform(
        tensor_b.device_view(),
        2,
        ElementInputB(4.f),
        ElementInputB(-4.f),
        0
    );

    cutlass::reference::device::TensorFillRandomUniform(
        tensor_c.device_view(),
        3,
        ElementAccumulator(4.f),
        ElementAccumulator(-4.f),
        0
    );

    MMAarguments mmaArg{
        problem_size,
        tensor_a.device_data(),
        tensor_b.device_data(),
        tensor_c.device_data(),
        tensor_mma_d.device_data()
    };

    GpuTimer mmatimer;
    mmatimer.start();
    launch_GEMM_MMA(mmaArg);
    mmatimer.stop();

    tensor_mma_d.sync_host();

    float mmarun_ms = mmatimer.elapsed_millis();

    double mmagflops = (double)problem_size.product()*2 / 1e9 / (mmarun_ms/1e3);

    std::cout << "Runtime(MMA): " << mmarun_ms << " ms" << std::endl;
    std::cout << "Gflops(MMA): " << mmagflops << std::endl; 

    // Initialize alpha and beta for dot product computation
    ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
    ElementComputeEpilogue beta = ElementComputeEpilogue(1);

    // Split K dimension into 1 partitions
    int split_k_slices = 1;

    // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
    // instantiated CUTLASS kernel
    typename Gemm::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                     tensor_a.device_ref(),  // <- reference to matrix A on device
                                     tensor_b.device_ref(),  // <- reference to matrix B on device
                                     tensor_c.device_ref(),  // <- reference to matrix C on device
                                     tensor_d.device_ref(),  // <- reference to matrix D on device
                                     {alpha, beta},          // <- tuple of alpha and beta
                                     split_k_slices};        // <- k-dimension split factor

    // Using the arguments, query for extra workspace required for matrix multiplication computation
    size_t workspace_size = Gemm::get_workspace_size(arguments);

    // Allocate workspace memory
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    // Instantiate CUTLASS kernel depending on templates
    Gemm gemm_op;

    // Check the problem size is supported or not 
    cutlass::Status status = gemm_op.can_implement(arguments);
    CUTLASS_CHECK(status);

    // Initialize CUTLASS kernel with arguments and workspace pointer
    status = gemm_op.initialize(arguments, workspace.get());
    CUTLASS_CHECK(status);

    GpuTimer cutimer;

    cutimer.start();
    status = gemm_op();
    cutimer.stop();
    CUTLASS_CHECK(status);

    tensor_d.sync_host();

    float curun_ms = cutimer.elapsed_millis();

    double cugflops = (double)problem_size.product()*2 / 1e9 / (curun_ms/1e3);

    std::cout << "Runtime(cutlass): " << curun_ms << " ms" << std::endl;
    std::cout << "Gflops(cutlass): " << cugflops << std::endl; 

    bool passed = cutlass::reference::host::TensorEquals(
        tensor_d.host_view(),
        tensor_mma_d.host_view()
    );

    if(passed) std::cout << "PASS\n";
    else {
        std::cout << "FAIL\n";
        tensor_a.sync_host();
        tensor_b.sync_host();
        tensor_c.sync_host();
        // std::cout << "A:\n" << tensor_a.host_view() << std::endl;
        // std::cout << "B:\n" << tensor_b.host_view() << std::endl;
        // std::cout << "C:\n" << tensor_c.host_view() << std::endl;
        // std::cout << "D(cutlass):\n" << tensor_d.host_view() << std::endl;
        // std::cout << "D(MMA)\n" << tensor_mma_d.host_view() << std::endl; 
        cutlass::reference::host::TensorSub<ElementOutput,LayoutOutput,ElementOutput,LayoutOutput,ElementOutput,LayoutOutput>(tensor_d.host_view(),tensor_mma_d.host_ref());
        std::cout << "diff=D(cutlass)-D(MMA):\n" << tensor_d.host_view() << std::endl;
    }
}
