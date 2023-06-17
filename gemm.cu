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

template<int NWARP_X=2,int NWARP_Y=2>
__device__ void loadtileA(MMAarguments &arg,ElementInputA *tileA,int idx){
    const int nrow = NWARP_Y * M;
    const int ncol = K;
    const int size = nrow * ncol;
    const int iter = (size+blockDim.x-1) / blockDim.x;
    // CUTLASS_PRAGMA_UNROLL
    for(int i=0;i<iter;i++){
        int tileIdx = threadIdx.x*iter + i;
        if(tileIdx>=size)return;
        int rowA = blockIdx.x*nrow + tileIdx/ncol ;
        int colA = idx*ncol + tileIdx%ncol ;
        tileA[tileIdx] = rowA<arg.problem_size.m() && colA<arg.problem_size.k() ? arg.A[rowA*arg.problem_size.k()+colA] : ElementInputA(0);
    }
}

template<int NWARP_X=2,int NWARP_Y=2>
__device__ void loadtileB(MMAarguments &arg,ElementInputB *tileB,int idx){
    const int nrow = K;
    const int ncol = NWARP_X * N;
    const int size = nrow * ncol;
    const int iter = (size+blockDim.x-1) / blockDim.x;
    // CUTLASS_PRAGMA_UNROLL
    for(int i=0;i<iter;i++){
        int tileIdx = threadIdx.x*iter + i;
        if(tileIdx>=size)return;
        int rowB = idx*nrow + tileIdx%nrow ;
        int colB = blockIdx.y*ncol + tileIdx/nrow ;
        tileB[tileIdx] = rowB<arg.problem_size.k() && colB<arg.problem_size.n() ? arg.B[colB*arg.problem_size.k()+rowB] : ElementInputB(0);
    }
}

template<int NWARP_X=2,int NWARP_Y=2>
__device__ void loadtileC(MMAarguments &arg,ElementAccumulator *tileC){
    const int nrow = NWARP_Y * M;
    const int ncol = NWARP_X * N; 
    const int size = nrow * ncol;
    const int iter = (size+blockDim.x-1) / blockDim.x;
    // CUTLASS_PRAGMA_UNROLL
    for(int i=0;i<iter;i++){
        int tileIdx = threadIdx.x*iter + i;
        if(tileIdx>=size)return;
        int rowC = blockIdx.x*nrow + tileIdx/ncol;
        int colC = blockIdx.y*ncol + tileIdx%ncol;
        tileC[tileIdx] = rowC<arg.problem_size.m() && colC<arg.problem_size.n() ? arg.C[rowC*arg.problem_size.n()+colC] : ElementAccumulator(0);
    }
}

// template<int NWARP_X=2,int NWARP_Y=2>
// __device__ void mvtile(MMAarguments &arg,ElementAccumulator *dst,ElementOutput *src){
//     const int size = NWARP_X*NWARP_Y*M*N;
//     const int iter = (size+blockDim.x-1) / blockDim.x;
//     // CUTLASS_PRAGMA_UNROLL
//     for(int i=0;i<iter;i++){
//         int tileIdx = threadIdx.x*iter + i;
//         if(tileIdx>=size)return;
//         dst[tileIdx] = src[tileIdx]; 
//     }
// }


template<int NWARP_X=2,int NWARP_Y=2>
__device__ void mmatile(MMAarguments &arg,ElementInputA *A,ElementInputB *B,ElementAccumulator *C,ElementOutput *D){
    const int warpidx = threadIdx.x / 32;
    const int rowwarp = warpidx / NWARP_X;
    const int colwarp = warpidx % NWARP_X;
    const int laneidx = threadIdx.x % 32;

    int a[4],b[2],cd[4];

    cd[0] = (rowwarp*M+laneidx/4)*NWARP_X*N+colwarp*N+laneidx%4*2;
    cd[1] = cd[0] + 1;
    cd[2] = cd[0] + 8*NWARP_X*N;
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

// __device__ void inittileD(ElementOutput *outD){
//     const int iter = 2*M*2*N / blockdim;
//     CUTLASS_PRAGMA_UNROLL
//     for(int i=0;i<iter;i++){
//         outD[i+threadIdx.x*iter] = 0;
//     }
// }

template<int NWARP_X=2,int NWARP_Y=2>
__device__ void storetile(MMAarguments &arg,ElementOutput *out){
    const int row = NWARP_Y * M;
    const int col = NWARP_X * N;
    const int size = row*col;
    const int iter = (size+blockDim.x-1)/blockDim.x;
    // CUTLASS_PRAGMA_UNROLL
    for(int i=0;i<iter;i++){
        int tileIdx = threadIdx.x*iter + i;
        if(tileIdx>=size)return;
        int rowD = blockIdx.x*row + tileIdx/col;
        int colD = blockIdx.y*col + tileIdx%col;
        if(rowD<arg.problem_size.m() && colD<arg.problem_size.n()) {
            arg.D[rowD*arg.problem_size.n()+colD] = out[tileIdx];
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

// __global__ void GEMM_MMA(MMAarguments arg){
//     __shared__ ElementInputA tileA[M*2*K]; 
//     __shared__ ElementInputB tileB[K*N*2]; 
//     __shared__ ElementOutput tileC[2*M*2*N];
//     __shared__ ElementOutput tileD[2*M*2*N];

//     const int iters = (arg.problem_size.k() + K - 1) / K;

//     loadtileC(arg,tileC);
    
//     for(int idx=0;idx<iters;idx++){
//         loadtileA(arg,tileA,idx);
//         loadtileB(arg,tileB,idx);
//         __syncthreads();
//         mmatile(arg,tileA,tileB,tileC,tileD);
//         __syncthreads();
//         mvtile(arg,tileC,tileD);
//     }
//     storetile(arg,tileD);
// }

template<int NWARP_X=2,int NWARP_Y=2>
__global__ void GEMM_MMA(MMAarguments arg){
    __shared__ ElementInputA tileA[NWARP_Y*M * K]; 
    __shared__ ElementInputB tileB[K * NWARP_X*N]; 
    __shared__ ElementOutput tileC[NWARP_Y*M * NWARP_X*N];

    const int iters = (arg.problem_size.k() + K - 1) / K;

    loadtileC<NWARP_X,NWARP_Y>(arg,tileC);
    
    for(int idx=0;idx<iters;idx++){
        loadtileA<NWARP_X,NWARP_Y>(arg,tileA,idx);
        loadtileB<NWARP_X,NWARP_Y>(arg,tileB,idx);
        __syncthreads();
        mmatile<NWARP_X,NWARP_Y>(arg,tileA,tileB,tileC,tileC);
        __syncthreads();
    }
    storetile<NWARP_X,NWARP_Y>(arg,tileC);
}

template<int NWARP_X=2,int NWARP_Y=2>
void launch_GEMM_MMA(MMAarguments arg){
    dim3 grid,block;
    grid.x = (arg.problem_size.m()+M*NWARP_Y-1)/(M*NWARP_Y);
    grid.y = (arg.problem_size.n()+N*NWARP_X-1)/(N*NWARP_X);
    grid.z = 1;

    block.x = NWARP_X * NWARP_Y * 32;
    block.y = 1;
    block.z = 1;

    GEMM_MMA<NWARP_X,NWARP_Y><<<grid,block>>>(arg);
}

// Create a tuple of problem size for matrix multiplication
cutlass::gemm::GemmCoord problem_size = {5120,4096,4096};

// Initialize tensors using CUTLASS helper functions
cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a;  // <- Create matrix A with dimensions M x K
cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_b;  // <- Create matrix B with dimensions K x N
cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_c;  // <- Create matrix C with dimensions M x N
cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_d;  

GpuTimer timer;

template<int i,int j>
void launch_GEMM_MMA_i_j(){
    cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_mma_d(problem_size.mn());
    std::string name_kernel = "MMA_tune_" + std::to_string(i) + "_" + std::to_string(j);
    std::string name_test = "cuMMA==MMA_tune_" + std::to_string(i) + "_" + std::to_string(j);
    timer.bind_run(name_kernel,[&]{
        MMAarguments mmaArg{
            problem_size,
            tensor_a.device_data(),
            tensor_b.device_data(),
            tensor_c.device_data(),
            tensor_mma_d.device_data()
        };
        launch_GEMM_MMA<i,j>(mmaArg);
    });
    
    timer.testEqual<ElementOutput,LayoutOutput>(name_test,tensor_d,tensor_mma_d);
}

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

    ////////////////////cuMMA////////////////////////////////

    timer.bind_run("cuMMA",[&]{
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

    //////////////////////GEMM_MMA_i_j///////////////////////
    {
        launch_GEMM_MMA_i_j<1,1>();
        launch_GEMM_MMA_i_j<1,2>();
        launch_GEMM_MMA_i_j<1,3>();
        launch_GEMM_MMA_i_j<1,4>();
        launch_GEMM_MMA_i_j<2,1>();
        launch_GEMM_MMA_i_j<2,2>();
        launch_GEMM_MMA_i_j<2,3>();
        launch_GEMM_MMA_i_j<2,4>();
        launch_GEMM_MMA_i_j<3,1>();
        launch_GEMM_MMA_i_j<3,2>();
        launch_GEMM_MMA_i_j<3,3>();
        launch_GEMM_MMA_i_j<3,4>();
        launch_GEMM_MMA_i_j<4,1>();
        launch_GEMM_MMA_i_j<4,2>();
        launch_GEMM_MMA_i_j<4,3>();
        launch_GEMM_MMA_i_j<4,4>(); 
    }
}
