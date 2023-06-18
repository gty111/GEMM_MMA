#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/kernel/default_gemm.h"
#include "cutlass/gemm/kernel/gemm.h"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_elementwise.h"
#include "cutlass/util/tensor_view_io.h"

#include <iostream>
#include <functional>

#define ENABLE_CUTLASS 1

/**
 * Panic wrapper for unwinding CUTLASS errors
 */
#define CUTLASS_CHECK(status)                                                                    \
  {                                                                                              \
    cutlass::Status error = status;                                                              \
    if (error != cutlass::Status::kSuccess) {                                                    \
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) << " at: " << __LINE__ \
                << std::endl;                                                                    \
      exit(EXIT_FAILURE);                                                                        \
    }                                                                                            \
  }


/**
 * Panic wrapper for unwinding CUDA runtime errors
 */
#define CUDA_CHECK(status)                                              \
  {                                                                     \
    cudaError_t error = status;                                         \
    if (error != cudaSuccess) {                                         \
      std::cerr << "Got bad cuda status: " << cudaGetErrorString(error) \
                << " at line: " << __LINE__ << std::endl;               \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  }

/**
 * GPU timer for recording the elapsed time across kernel(s) launched in GPU stream
 */
struct GpuTimer
{
    cudaStream_t _stream_id;
    cudaEvent_t _start;
    cudaEvent_t _stop;
    cutlass::gemm::GemmCoord problem_size={0,0,0};

    /// Constructor
    GpuTimer() : _stream_id(0)
    {
        CUDA_CHECK(cudaEventCreate(&_start));
        CUDA_CHECK(cudaEventCreate(&_stop));
    }

    /// Destructor
    ~GpuTimer()
    {
        CUDA_CHECK(cudaEventDestroy(_start));
        CUDA_CHECK(cudaEventDestroy(_stop));
    }

    void set(cutlass::gemm::GemmCoord &_problem_size){
        problem_size = _problem_size;
    }

    /// Start the timer for a given stream (defaults to the default stream)
    void start(cudaStream_t stream_id = 0)
    {
        _stream_id = stream_id;
        CUDA_CHECK(cudaEventRecord(_start, _stream_id));
    }

    /// Stop the timer
    void stop()
    {
        CUDA_CHECK(cudaEventRecord(_stop, _stream_id));
    }

    /// Return the elapsed time (in milliseconds)
    float elapsed_millis()
    {
        float elapsed = 0.0;
        CUDA_CHECK(cudaEventSynchronize(_stop));
        CUDA_CHECK(cudaEventElapsedTime(&elapsed, _start, _stop));
        return elapsed;
    }

    void bind_run(std::string name,const std::function<void()> &kernel,int test_time=10){
        float run_ms = 0;
        for(int i=0;i<test_time;i++){
            start();
            kernel();
            stop();
            run_ms += elapsed_millis();
        }
        run_ms /= (float)test_time;
        if(problem_size.product()){
            double gflops = (double)problem_size.product()*2 / 1e9 / (run_ms/1e3);
            std::printf("[%20s] Runtime: %f(ms) Gflops: %f\n",name.c_str(),run_ms,gflops);
        }else{
            std::printf("[%20s] Runtime: %f(ms)\n",name.c_str(),run_ms);
        }
    }

    template<typename E,typename L>
    void testEqual(std::string name,cutlass::HostTensor<E,L> &a,cutlass::HostTensor<E,L> &b,bool ifprint=0){
        a.sync_host();
        b.sync_host();
        bool passed = cutlass::reference::host::TensorEquals(
            a.host_view(),
            b.host_view()
        );

        if(passed) std::printf("[%20s] PASS\n",name.c_str());
        else {
            std::printf("[%20s] FAIL\n",name.c_str());
            if(ifprint){
                std::cout << "[a]:\n" << a.host_view() << std::endl;
                std::cout << "[b]:\n" << b.host_view() << std::endl;
                cutlass::reference::host::TensorSub<E,L,E,L,E,L>(a.host_view(),b.host_ref());
                std::cout << "diff:\n" << a.host_view() << std::endl;
            }
        }
    }

    template<typename E,typename L>
    void printTensor(std::string name,cutlass::HostTensor<E,L> &a){
        a.sync_host();
        std::cout << name << ":\n" << a.host_view() << std::endl;
    }
};

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

__device__ void loadtileC(MMAarguments &arg,ElementOutput *C){
    // iter = 128 * 64 / 128
    for(int i=0;i<64;i++){
        int tileIdx = threadIdx.x*64 + i;
        int rowC = tileIdx / 64 + blockIdx.y * 128;
        int colC = tileIdx % 64 + blockIdx.x * 64;

        C[tileIdx] =  rowC<arg.problem_size.m()&&colC<arg.problem_size.n() ? arg.C[rowC*arg.problem_size.n()+colC] : ElementOutput(0);
    }
}

__device__ void loadtileA(MMAarguments &arg,ElementInputA *A,int idx){
    // iter = 128 * 8 / 128
    for(int i=0;i<8;i++){
        int tileIdx = threadIdx.x*8 + i;
        int rowA = tileIdx/K + blockIdx.y * 128;
        int colA = tileIdx%K + idx*K; 

        A[tileIdx] = rowA<arg.problem_size.m()&&colA<arg.problem_size.k() ? arg.A[rowA*arg.problem_size.k()+colA] : ElementInputA(0);
    }
}

__device__ void loadtileB(MMAarguments &arg,ElementInputB *B,int idx){
    // iter = 64 * 8 / 128
    for(int i=0;i<4;i++){
        int tileIdx = threadIdx.x*4 + i;
        int rowB = idx*K + tileIdx%K;
        int colB = blockIdx.x*64 + tileIdx/K;

        B[tileIdx] = rowB<arg.problem_size.k()&&colB<arg.problem_size.n() ? arg.B[colB*arg.problem_size.k()+rowB] : ElementInputB(0);
    }
}

__device__ void storetile(MMAarguments &arg,ElementOutput *D){
    // iter = 128 * 64 / 128
    for(int i=0;i<64;i++){
        int tileIdx = threadIdx.x*64 + i;
        int rowD = tileIdx / 64 + blockIdx.y * 128;
        int colD = tileIdx % 64 + blockIdx.x * 64;

        if(rowD<arg.problem_size.m()&&colD<arg.problem_size.n()){
            arg.D[rowD*arg.problem_size.n()+colD] = D[tileIdx];
        }
    }
}

__device__ bool test(int a0,int b0,int a1,int b1){
    return a0<a1 && b0 < b1;
}

__device__ void loadtileC_vec(MMAarguments &arg,ElementOutput *C){
    // iter = 128 * 64 / 128
    for(int i=0;i<16;i++){
        int tileIdx = threadIdx.x*64 + i*4;
        int rowC_0 = tileIdx / 64 + blockIdx.y * 128;
        int colC_0 = tileIdx % 64 + blockIdx.x * 64;
        int rowC_1 = (tileIdx+1) / 64 + blockIdx.y * 128;
        int colC_1 = (tileIdx+1) % 64 + blockIdx.x * 64;
        int rowC_2 = (tileIdx+2) / 64 + blockIdx.y * 128;
        int colC_2 = (tileIdx+2) % 64 + blockIdx.x * 64;
        int rowC_3 = (tileIdx+3) / 64 + blockIdx.y * 128;
        int colC_3 = (tileIdx+3) % 64 + blockIdx.x * 64;

        bool test0 = test(rowC_0,colC_0,arg.problem_size.m(),arg.problem_size.n()); 
        bool test1 = test(rowC_1,colC_1,arg.problem_size.m(),arg.problem_size.n());
        bool test2 = test(rowC_2,colC_2,arg.problem_size.m(),arg.problem_size.n());
        bool test3 = test(rowC_3,colC_3,arg.problem_size.m(),arg.problem_size.n());

        if(test0&&test1&&test2&&test3)
            *reinterpret_cast<float4*>(&C[tileIdx]) = *reinterpret_cast<float4*>(&arg.C[rowC_0*arg.problem_size.n()+colC_0]);
        else{
            C[tileIdx]   = test0 ? arg.C[rowC_0*arg.problem_size.n()+colC_0] : ElementOutput(0);
            C[tileIdx+1] = test1 ? arg.C[rowC_1*arg.problem_size.n()+colC_1] : ElementOutput(0);
            C[tileIdx+2] = test2 ? arg.C[rowC_2*arg.problem_size.n()+colC_2] : ElementOutput(0);
            C[tileIdx+3] = test3 ? arg.C[rowC_3*arg.problem_size.n()+colC_3] : ElementOutput(0);
        }
    }
}

__device__ void loadtileA_vec(MMAarguments &arg,ElementInputA *A,int idx){
    // iter = 128 * 8 / 128
    for(int i=0;i<2;i++){
        int tileIdx = threadIdx.x*8 + i*4;
        int rowA_0 = tileIdx/K + blockIdx.y * 128;
        int colA_0 = tileIdx%K + idx*K; 
        int rowA_1 = (tileIdx+1)/K + blockIdx.y * 128;
        int colA_1 = (tileIdx+1)%K + idx*K; 
        int rowA_2 = (tileIdx+2)/K + blockIdx.y * 128;
        int colA_2 = (tileIdx+2)%K + idx*K; 
        int rowA_3 = (tileIdx+3)/K + blockIdx.y * 128;
        int colA_3 = (tileIdx+3)%K + idx*K; 

        bool test0 = test(rowA_0,colA_0,arg.problem_size.m(),arg.problem_size.k()); 
        bool test1 = test(rowA_1,colA_1,arg.problem_size.m(),arg.problem_size.k());
        bool test2 = test(rowA_2,colA_2,arg.problem_size.m(),arg.problem_size.k());
        bool test3 = test(rowA_3,colA_3,arg.problem_size.m(),arg.problem_size.k());

        if(test0&&test1&&test2&&test3)
            *reinterpret_cast<float4*>(&A[tileIdx]) = *reinterpret_cast<float4*>(&arg.A[rowA_0*arg.problem_size.k()+colA_0]);
        else{
            A[tileIdx]   = test0 ? arg.A[rowA_0*arg.problem_size.k()+colA_0] : ElementInputA(0);
            A[tileIdx+1] = test1 ? arg.A[rowA_1*arg.problem_size.k()+colA_1] : ElementInputA(0);
            A[tileIdx+2] = test2 ? arg.A[rowA_2*arg.problem_size.k()+colA_2] : ElementInputA(0);
            A[tileIdx+3] = test3 ? arg.A[rowA_3*arg.problem_size.k()+colA_3] : ElementInputA(0);
        }
    }
}

__device__ void loadtileB_vec(MMAarguments &arg,ElementInputB *B,int idx){
    // iter = 64 * 8 / 128
    for(int i=0;i<1;i++){
        int tileIdx = threadIdx.x*4 + i*4;
        int rowB_0 = idx*K + tileIdx%K;
        int colB_0 = blockIdx.x*64 + tileIdx/K;
        int rowB_1 = idx*K + (tileIdx+1)%K;
        int colB_1 = blockIdx.x*64 + (tileIdx+1)/K;
        int rowB_2 = idx*K + (tileIdx+2)%K;
        int colB_2 = blockIdx.x*64 + (tileIdx+2)/K;
        int rowB_3 = idx*K + (tileIdx+3)%K;
        int colB_3 = blockIdx.x*64 + (tileIdx+3)/K;

        bool test0 = test(rowB_0,colB_0,arg.problem_size.k(),arg.problem_size.n()); 
        bool test1 = test(rowB_1,colB_1,arg.problem_size.k(),arg.problem_size.n());
        bool test2 = test(rowB_2,colB_2,arg.problem_size.k(),arg.problem_size.n());
        bool test3 = test(rowB_3,colB_3,arg.problem_size.k(),arg.problem_size.n());

        if(test0&&test1&&test2&&test3)
            *reinterpret_cast<float4*>(&B[tileIdx]) = *reinterpret_cast<float4*>(&arg.B[colB_0*arg.problem_size.k()+rowB_0]);
        else{
            B[tileIdx]   = test0 ? arg.B[colB_0*arg.problem_size.k()+rowB_0] : ElementInputB(0);
            B[tileIdx+1] = test1 ? arg.B[colB_1*arg.problem_size.k()+rowB_1] : ElementInputB(0);
            B[tileIdx+2] = test2 ? arg.B[colB_2*arg.problem_size.k()+rowB_2] : ElementInputB(0);
            B[tileIdx+3] = test3 ? arg.B[colB_3*arg.problem_size.k()+rowB_3] : ElementInputB(0);
        }
    }
}

__device__ void storetile_vec(MMAarguments &arg,ElementOutput *D){
    // iter = 128 * 64 / 128
    for(int i=0;i<16;i++){
        int tileIdx = threadIdx.x*64 + i*4;
        int rowD_0 = tileIdx / 64 + blockIdx.y * 128;
        int colD_0 = tileIdx % 64 + blockIdx.x * 64;
        int rowD_1 = (tileIdx+1) / 64 + blockIdx.y * 128;
        int colD_1 = (tileIdx+1) % 64 + blockIdx.x * 64;
        int rowD_2 = (tileIdx+2) / 64 + blockIdx.y * 128;
        int colD_2 = (tileIdx+2) % 64 + blockIdx.x * 64;
        int rowD_3 = (tileIdx+3) / 64 + blockIdx.y * 128;
        int colD_3 = (tileIdx+3) % 64 + blockIdx.x * 64;

        bool test0 = test(rowD_0,colD_0,arg.problem_size.m(),arg.problem_size.n()); 
        bool test1 = test(rowD_1,colD_1,arg.problem_size.m(),arg.problem_size.n());
        bool test2 = test(rowD_2,colD_2,arg.problem_size.m(),arg.problem_size.n());
        bool test3 = test(rowD_3,colD_3,arg.problem_size.m(),arg.problem_size.n());

        if(test0&&test1&&test2&&test3)
            *reinterpret_cast<float4*>(&arg.D[rowD_0*arg.problem_size.n()+colD_0]) = *reinterpret_cast<float4*>(&D[tileIdx]);
        else{
            if(test0)
                arg.D[rowD_0*arg.problem_size.n()+colD_0] = D[tileIdx];
            if(test1)
                arg.D[rowD_1*arg.problem_size.n()+colD_1] = D[tileIdx+1];
            if(test2)
                arg.D[rowD_2*arg.problem_size.n()+colD_2] = D[tileIdx+2];
            if(test3)
                arg.D[rowD_3*arg.problem_size.n()+colD_3] = D[tileIdx+3];
        }
    }
}

__device__ void mma_tile(MMAarguments &arg,ElementInputA *A,ElementInputB *B,ElementAccumulator *C,ElementOutput *D){
    const int warpidx = threadIdx.x / 32;
    const int rowwarp = warpidx / 2;
    const int colwarp = warpidx % 2;
    const int laneidx = threadIdx.x % 32;

    int a[4],b[2],cd[4];

    for(int tileidx=0;tileidx<16;tileidx++){
        int rowtile = tileidx / 4;
        int coltile = tileidx % 4;
        cd[0] = (rowwarp*64+rowtile*M+laneidx/4)*64 + colwarp*32 + coltile*N + laneidx%4*2;
        cd[1] = cd[0] + 1;
        cd[2] = cd[0] + 8*64;
        cd[3] = cd[2] + 1;

        a[0] = (rowwarp*64+rowtile*M+laneidx/4)*K + laneidx%4;
        a[1] = a[0] + 8*K;
        a[2] = a[0] + 4;
        a[3] = a[1] + 4;

        b[0] = (colwarp*32+coltile*N+laneidx/4)*K + laneidx%4;
        b[1] = b[0] + 4;

        asm volatile(
            "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
            : 
            "=f"(C[cd[0]]),  // D[0]
            "=f"(C[cd[1]]),  // D[1]
            "=f"(C[cd[2]]),  // D[2]
            "=f"(C[cd[3]])   // D[3]
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
    __shared__ ElementInputA tileA[128*8];
    __shared__ ElementInputB tileB[8*64];
    __shared__ ElementOutput tileC[128*64];

    const int iters = (arg.problem_size.k() + K - 1) / K;
    
    loadtileC(arg,tileC);
    
    for(int i=0;i<iters;i++){
        loadtileA(arg,tileA,i);
        loadtileB(arg,tileB,i);

        __syncthreads();
        mma_tile(arg,tileA,tileB,tileC,tileC);
        __syncthreads();
    }

    storetile(arg,tileC);
}

__global__ void GEMM_MMA_vec(MMAarguments arg){
    __shared__ ElementInputA tileA[128*8];
    __shared__ ElementInputB tileB[8*64];
    __shared__ ElementOutput tileC[128*64];

    const int iters = (arg.problem_size.k() + K - 1) / K;
    
    loadtileC_vec(arg,tileC);

    for(int i=0;i<iters;i++){
        loadtileA_vec(arg,tileA,i);
        loadtileB_vec(arg,tileB,i);
        __syncthreads();
        mma_tile(arg,tileA,tileB,tileC,tileC);
        __syncthreads();
    }

    storetile_vec(arg,tileC);
}

void launch_GEMM_MMA(MMAarguments &arg){
    dim3 grid,block;
    // threadblockShape 128 64 8
    // warpShape 64 32 8
    // every block has 4 warps

    grid.x = (arg.problem_size.n()+64-1)/64;
    grid.y = (arg.problem_size.m()+128-1)/128;
    grid.z = 1;

    block.x = 128;
    block.y = 1;
    block.z = 1;

    GEMM_MMA<<<grid,block>>>(arg);
}

void launch_GEMM_MMA_vec(MMAarguments &arg){
    dim3 grid,block;
    // threadblockShape 128 64 8
    // warpShape 64 32 8
    // every block has 4 warps

    grid.x = (arg.problem_size.n()+64-1)/64;
    grid.y = (arg.problem_size.m()+128-1)/128;
    grid.z = 1;

    block.x = 128;
    block.y = 1;
    block.z = 1;

    GEMM_MMA_vec<<<grid,block>>>(arg);
}

// Create a tuple of problem size for matrix multiplication
cutlass::gemm::GemmCoord problem_size = {5120,4096,4096};

// Initialize tensors using CUTLASS helper functions
cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a;  // <- Create matrix A with dimensions M x K
cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_b;  // <- Create matrix B with dimensions K x N
cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_c;  // <- Create matrix C with dimensions M x N
cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_d;  

GpuTimer timer;

int main(int argc,char **argv){
    //////////////////////////INIT////////////////////////////////

    if(argc>=2)problem_size.m()=atoi(argv[1]);
    if(argc>=3)problem_size.n()=atoi(argv[2]);
    if(argc>=4)problem_size.k()=atoi(argv[3]);

    printf("[%20s] (%d,%d,%d)\n","problem size",problem_size.m(),problem_size.n(),problem_size.k());

    tensor_a.resize(problem_size.mk());
    tensor_b.resize(problem_size.kn());
    tensor_c.resize(problem_size.mn());
    tensor_d.resize(problem_size.mn());

    timer.set(problem_size);
    
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

    ////////////////////cutlassMMA////////////////////////////////
  #if ENABLE_CUTLASS
    timer.bind_run("cutlassMMA",[&]{
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

        // Launch CUTLASS kernel
        status = gemm_op();
        CUTLASS_CHECK(status);
    });
  #endif
    //////////////////////GEMM_MMA///////////////////////
    {   
        #if ENABLE_CUTLASS
        cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_mma_d(problem_size.mn());
        #endif

        timer.bind_run("MMA_base",[&]{
            MMAarguments mmaArg{
                problem_size,
                tensor_a.device_data(),
                tensor_b.device_data(),
                tensor_c.device_data(),
                #if ENABLE_CUTLASS
                tensor_mma_d.device_data()
                #else
                tensor_d.device_data()
                #endif
            };
            launch_GEMM_MMA(mmaArg);
        });
        
        #if ENABLE_CUTLASS
        timer.testEqual<ElementOutput,LayoutOutput>("MMA_base==ref",tensor_d,tensor_mma_d);
        #endif
    }

    {
        cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_mma_d(problem_size.mn());

        timer.bind_run("MMA_vec",[&]{
            MMAarguments mmaArg{
                problem_size,
                tensor_a.device_data(),
                tensor_b.device_data(),
                tensor_c.device_data(),
                tensor_mma_d.device_data()
            };
            launch_GEMM_MMA_vec(mmaArg);
        });
        
        timer.testEqual<ElementOutput,LayoutOutput>("MMA_vec==ref",tensor_d,tensor_mma_d);
    }
}