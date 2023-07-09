#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/kernel/default_gemm.h"
#include "cutlass/gemm/kernel/gemm.h"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_elementwise.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/command_line.h"

#include <iostream>
#include <functional>

#define DIV(x,y) ((x)+(y)-1)/(y)

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

    void bind_run(std::string name,const std::function<void()> &kernel,int test_time=20){
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

// Command line options parsing
struct Options {

  bool help;

  cutlass::gemm::GemmCoord problem_size;
  int iterations;
  bool ifprint;
  
  Options():
    help(false),
    problem_size({5120, 4096, 4096}),
    iterations(20),
    ifprint(false)
    { }

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
    }

    cmd.get_cmd_line_argument("m", problem_size.m());
    cmd.get_cmd_line_argument("n", problem_size.n());
    cmd.get_cmd_line_argument("k", problem_size.k());
    
    cmd.get_cmd_line_argument("iterations", iterations);
    cmd.get_cmd_line_argument("print",ifprint);

  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "Options:\n\n"
        << "  --help                      If specified, displays this usage statement.\n\n"
        << "  --m=<int>                   GEMM M dimension\n"
        << "  --n=<int>                   GEMM N dimension\n"
        << "  --k=<int>                   GEMM K dimension\n"
        << "  --iterations=<int>          Number of profiling iterations to perform\n"
        << "  --print=<bool>              print debug info\n\n";

    return out;
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

// const int M = ShapeMMAOp::kM;
// const int N = ShapeMMAOp::kN;
// const int K = ShapeMMAOp::kK;

const int logtile = 2; // for threadblock swizzle

struct MMAarguments{
    cutlass::gemm::GemmCoord problem_size;
    ElementInputA *A;
    ElementInputB *B;
    ElementAccumulator *C;
    ElementOutput *D;
};

struct Index{
    int blockIdx_x,blockIdx_y;
    int warpidx;
    int laneidx;
    int rowwarp;
    int colwarp;
    int rowA[2][2][4];
    int colA[2][2][4];
    int rowB[2][2][4];
    int colB[2][2][4];
    int tileidx[4];
    int a[2][4];
    int b[2][4];

    __device__ Index(){
        warpidx = threadIdx.x / 32;
        laneidx = threadIdx.x % 32;
        rowwarp = warpidx / 2;
        colwarp = warpidx % 2;

        blockIdx_x = blockIdx.x >> logtile;
        blockIdx_y = (blockIdx.y << logtile) + ((blockIdx.x) & ((1 << (logtile)) - 1));

        for(int i=0;i<4;i++){
            tileidx[i] = laneidx/2*8 + (((laneidx>>3)^laneidx)&1)*4 + warpidx*128 + i*128*4;
        }

        for(int k=0;k<2;k++){
            for(int i=0;i<2;i++){
                for(int j=0;j<4;j++){
                    rowA[k][i][j] = laneidx/2 + warpidx*16 + i*64 + blockIdx_y * 128;
                    colA[k][i][j] = laneidx%2*4 + j + k*8;
                }
            }
        }
        
        for(int k=0;k<2;k++){
            for(int i=0;i<2;i++){
                for(int j=0;j<4;j++){
                    rowB[k][i][j] = laneidx%2*4 + j + k*8;
                    colB[k][i][j] = laneidx/2 + warpidx*16 + i*64 + blockIdx_x * 128;
                }
            }
        }
        
        for(int k=0;k<2;k++){
            for(int i=0;i<4;i++){
                a[k][i] = (rowwarp*64+i*16+laneidx%16)*8+(((laneidx>>2)^(laneidx>>4))&1)*4+k*128*8;
                b[k][i] = (colwarp*64+i*16+laneidx%8+laneidx/16*8)*8+(((laneidx>>2)^(laneidx>>3))&1)*4+k*128*8;
            }
        }
        
    }
};

template<typename T>
__device__ void printvar(T var){
    if(threadIdx.x!=0||blockIdx.x!=0||blockIdx.y!=0)return;
    printf("%d\n",(int)var);
}

template<typename T>
__device__ void printtile(T *arr,int row,int col,bool rowMajor){
    if(threadIdx.x!=0||blockIdx.x!=0||blockIdx.y!=0)return;
    printf("---------------------\n");
    for(int i=0;i<row;i++){
        for(int j=0;j<col;j++){
            if(rowMajor)
                printf("%d, ",(int)arr[i*col+j]);
            else
                printf("%d, ",(int)arr[j*row+i]);
        }
        printf("\n");
    }
}

__device__ bool test(int a0,int b0,int a1,int b1){
    return a0<a1 && b0 < b1;
}

__device__ void loadtileC(MMAarguments &arg,ElementOutput *C_fragemnt1,ElementOutput *C_fragemnt2,Index &index){
    int rowtile,coltile,rowC,colC;
    bool test0,test1;
    for(int i=0;i<16;i++){
        rowtile = i / 8;
        coltile = i % 8;
        for(int j=0;j<2;j++){
            rowC = index.blockIdx_y*128 + index.rowwarp*64 + rowtile*16 + index.laneidx/4 + j*8;
            colC = index.blockIdx_x*128 + index.colwarp*64 + coltile*8  + index.laneidx%4*2;

            test0 = test(rowC,colC,arg.problem_size.m(),arg.problem_size.n());
            test1 = test(rowC,colC+1,arg.problem_size.m(),arg.problem_size.n());

            if(test1){   
                *reinterpret_cast<float2*>(&C_fragemnt1[i*4+j*2]) = *reinterpret_cast<float2*>(&arg.C[rowC*arg.problem_size.n()+colC]) ; 
            }else{
                C_fragemnt1[i*4+j*2]   = test0 ? arg.C[rowC*arg.problem_size.n()+colC] : ElementOutput(0); 
                C_fragemnt1[i*4+j*2+1] = test1 ? arg.C[rowC*arg.problem_size.n()+colC+1] : ElementOutput(0);
            }
        }   
    }

    for(int i=0;i<16;i++){
        rowtile = i / 8 + 2;
        coltile = i % 8;
        for(int j=0;j<2;j++){
            rowC = index.blockIdx_y*128 + index.rowwarp*64 + rowtile*16 + index.laneidx/4 + j*8;
            colC = index.blockIdx_x*128 + index.colwarp*64 + coltile*8  + index.laneidx%4*2;

            test0 = test(rowC,colC,arg.problem_size.m(),arg.problem_size.n());
            test1 = test(rowC,colC+1,arg.problem_size.m(),arg.problem_size.n());

            if(test1){   
                *reinterpret_cast<float2*>(&C_fragemnt2[i*4+j*2]) = *reinterpret_cast<float2*>(&arg.C[rowC*arg.problem_size.n()+colC]) ; 
            }else{
                C_fragemnt2[i*4+j*2]   = test0 ? arg.C[rowC*arg.problem_size.n()+colC] : ElementOutput(0); 
                C_fragemnt2[i*4+j*2+1] = test1 ? arg.C[rowC*arg.problem_size.n()+colC+1] : ElementOutput(0);
            }
        }   
    }
}

__device__ void storetile(MMAarguments &arg,ElementOutput *C_fragment1,ElementOutput *C_fragment2,Index &index){
    int rowtile,coltile,rowC,colC;
    bool test0,test1;

    for(int i=0;i<16;i++){
        rowtile = i / 8;
        coltile = i % 8;
        for(int j=0;j<2;j++){
            rowC = index.blockIdx_y*128 + index.rowwarp*64 + rowtile*16 + index.laneidx/4 + j*8 ;
            colC = index.blockIdx_x*128 + index.colwarp*64 + coltile*8  + index.laneidx%4*2;

            test0 = test(rowC,colC,arg.problem_size.m(),arg.problem_size.n());
            test1 = test(rowC,colC+1,arg.problem_size.m(),arg.problem_size.n());

            if(test1){   
                *reinterpret_cast<float2*>(&arg.D[rowC*arg.problem_size.n()+colC]) = *reinterpret_cast<float2*>(&C_fragment1[i*4+j*2]) ; 
            }else{
                if(test0)
                    arg.D[rowC*arg.problem_size.n()+colC] = C_fragment1[i*4+j*2];
                if(test1)
                    arg.D[rowC*arg.problem_size.n()+colC+1] = C_fragment1[i*4+j*2+1];
            }
        }   
    }

    for(int i=0;i<16;i++){
        rowtile = i / 8 + 2;
        coltile = i % 8;
        for(int j=0;j<2;j++){
            rowC = index.blockIdx_y*128 + index.rowwarp*64 + rowtile*16 + index.laneidx/4 + j*8 ;
            colC = index.blockIdx_x*128 + index.colwarp*64 + coltile*8  + index.laneidx%4*2;

            test0 = test(rowC,colC,arg.problem_size.m(),arg.problem_size.n());
            test1 = test(rowC,colC+1,arg.problem_size.m(),arg.problem_size.n());

            if(test1){   
                *reinterpret_cast<float2*>(&arg.D[rowC*arg.problem_size.n()+colC]) = *reinterpret_cast<float2*>(&C_fragment2[i*4+j*2]) ; 
            }else{
                if(test0)
                    arg.D[rowC*arg.problem_size.n()+colC] = C_fragment2[i*4+j*2];
                if(test1)
                    arg.D[rowC*arg.problem_size.n()+colC+1] = C_fragment2[i*4+j*2+1];
            }
        }   
    }
}

__device__ void ldsA(ElementInputA *As,ElementInputA *Ar,Index &index){
    for(int k=0;k<2;k++){
        for(int i=0;i<4;i++){
            asm volatile(
                "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];" 
                : 
                "=r"(*(uint32_t*)&Ar[k*16+i*4]), 
                "=r"(*(uint32_t*)&Ar[k*16+i*4+1]), 
                "=r"(*(uint32_t*)&Ar[k*16+i*4+2]), 
                "=r"(*(uint32_t*)&Ar[k*16+i*4+3])
                : 
                "r"((uint32_t)__cvta_generic_to_shared(&As[index.a[k][i]])) 
            );
        }
    }
}

__device__ void ldsB(ElementInputB *Bs,ElementInputB *Br,Index &index){

    for(int k=0;k<2;k++){
        for(int i=0;i<4;i++){
            asm volatile(
                "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];" 
                : 
                "=r"(*(uint32_t*)&Br[k*16+i*4]), 
                "=r"(*(uint32_t*)&Br[k*16+i*4+1]), 
                "=r"(*(uint32_t*)&Br[k*16+i*4+2]), 
                "=r"(*(uint32_t*)&Br[k*16+i*4+3])
                :
                "r"((uint32_t)__cvta_generic_to_shared(&Bs[index.b[k][i]])) 
            );
        }
    }
    
}

__device__ void mma_tile(MMAarguments &arg,ElementInputA *A_fragment,ElementInputB *B_fragment,ElementOutput *C_fragment1,ElementOutput *C_fragment2){

    for(int k=0;k<2;k++)
    for(int i=0;i<16;i++){
        asm volatile(
            "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
            : 
            "=f"(C_fragment1[i*4+0]),  // D[0]
            "=f"(C_fragment1[i*4+1]),  // D[1]
            "=f"(C_fragment1[i*4+2]),  // D[2]
            "=f"(C_fragment1[i*4+3])   // D[3]
            : 
            "r"(*reinterpret_cast<uint32_t const *>(&A_fragment[i/8*4+0+k*16])),   // A[0]
            "r"(*reinterpret_cast<uint32_t const *>(&A_fragment[i/8*4+1+k*16])),   // A[1]
            "r"(*reinterpret_cast<uint32_t const *>(&A_fragment[i/8*4+2+k*16])),   // A[2]
            "r"(*reinterpret_cast<uint32_t const *>(&A_fragment[i/8*4+3+k*16])),   // A[3]
            "r"(*reinterpret_cast<uint32_t const *>(&B_fragment[i%8*2+0+k*16])),   // B[0]
            "r"(*reinterpret_cast<uint32_t const *>(&B_fragment[i%8*2+1+k*16])),   // B[1]
            "f"(C_fragment1[i*4+0]),   // C[0]
            "f"(C_fragment1[i*4+1]),   // C[1]
            "f"(C_fragment1[i*4+2]),   // C[2]
            "f"(C_fragment1[i*4+3])    // C[3]
        );
    }

    for(int k=0;k<2;k++)
    for(int i=0;i<16;i++){
        asm volatile(
            "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
            : 
            "=f"(C_fragment2[i*4+0]),  // D[0]
            "=f"(C_fragment2[i*4+1]),  // D[1]
            "=f"(C_fragment2[i*4+2]),  // D[2]
            "=f"(C_fragment2[i*4+3])   // D[3]
            : 
            "r"(*reinterpret_cast<uint32_t const *>(&A_fragment[i/8*4+0+8+k*16])),   // A[0]
            "r"(*reinterpret_cast<uint32_t const *>(&A_fragment[i/8*4+1+8+k*16])),   // A[1]
            "r"(*reinterpret_cast<uint32_t const *>(&A_fragment[i/8*4+2+8+k*16])),   // A[2]
            "r"(*reinterpret_cast<uint32_t const *>(&A_fragment[i/8*4+3+8+k*16])),   // A[3]
            "r"(*reinterpret_cast<uint32_t const *>(&B_fragment[i%8*2+0+k*16])),   // B[0]
            "r"(*reinterpret_cast<uint32_t const *>(&B_fragment[i%8*2+1+k*16])),   // B[1]
            "f"(C_fragment2[i*4+0]),   // C[0]
            "f"(C_fragment2[i*4+1]),   // C[1]
            "f"(C_fragment2[i*4+2]),   // C[2]
            "f"(C_fragment2[i*4+3])    // C[3]
        );
    }
}

#define LDGSTSCG(smem_addr,glb_addr,nbytes) asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" \
                                                            :: "r"((uint32_t)__cvta_generic_to_shared(smem_addr)), \
                                                            "l"(glb_addr), \
                                                            "n"(nbytes))

#define LDGSTSCA(smem_addr,glb_addr,nbytes,src_in_bytes) asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %3;\n" \
                                                            :: "r"((uint32_t)__cvta_generic_to_shared(smem_addr)), \
                                                            "l"(glb_addr), \
                                                            "n"(nbytes), \
                                                            "r"(src_in_bytes))

__device__ void loadtileA(MMAarguments &arg,ElementInputA *A,Index &index){

    bool flag[4];

    for(int k=0;k<2;k++){
        for(int i=0;i<2;i++){
            flag[0] = test(index.rowA[k][i][0],index.colA[k][i][0],arg.problem_size.m(),arg.problem_size.k()); 
            flag[1] = test(index.rowA[k][i][1],index.colA[k][i][1],arg.problem_size.m(),arg.problem_size.k());
            flag[2] = test(index.rowA[k][i][2],index.colA[k][i][2],arg.problem_size.m(),arg.problem_size.k());
            flag[3] = test(index.rowA[k][i][3],index.colA[k][i][3],arg.problem_size.m(),arg.problem_size.k());

            if(flag[3]){
                LDGSTSCG(&A[index.tileidx[k*2+i]],&arg.A[index.rowA[k][i][0]*arg.problem_size.k()+index.colA[k][i][0]],16);
            }else{
                LDGSTSCA(&A[index.tileidx[k*2+i]+0],&arg.A[index.rowA[k][i][0]*arg.problem_size.k()+index.colA[k][i][0]],4,flag[0]*4);
                LDGSTSCA(&A[index.tileidx[k*2+i]+1],&arg.A[index.rowA[k][i][1]*arg.problem_size.k()+index.colA[k][i][1]],4,flag[1]*4);
                LDGSTSCA(&A[index.tileidx[k*2+i]+2],&arg.A[index.rowA[k][i][2]*arg.problem_size.k()+index.colA[k][i][2]],4,flag[2]*4);
                LDGSTSCA(&A[index.tileidx[k*2+i]+3],&arg.A[index.rowA[k][i][3]*arg.problem_size.k()+index.colA[k][i][3]],4,flag[3]*4);
            }
        }
    }

    for(int k=0;k<2;k++){
        for(int i=0;i<2;i++){
            for(int j=0;j<4;j++){
                index.colA[k][i][j] += 16;
            }
        }
    }
}

__device__ void loadtileB(MMAarguments &arg,ElementInputB *B,Index &index){

    bool flag[4];

    for(int k=0;k<2;k++){
        for(int i=0;i<2;i++){
            flag[0] = test(index.rowB[k][i][0],index.colB[k][i][0],arg.problem_size.k(),arg.problem_size.n()); 
            flag[1] = test(index.rowB[k][i][1],index.colB[k][i][1],arg.problem_size.k(),arg.problem_size.n());
            flag[2] = test(index.rowB[k][i][2],index.colB[k][i][2],arg.problem_size.k(),arg.problem_size.n());
            flag[3] = test(index.rowB[k][i][3],index.colB[k][i][3],arg.problem_size.k(),arg.problem_size.n());

            if(flag[3]){
                LDGSTSCG(&B[index.tileidx[k*2+i]],&arg.B[index.colB[k][i][0]*arg.problem_size.k()+index.rowB[k][i][0]],16);
            }else{
                LDGSTSCA(&B[index.tileidx[k*2+i]+0],&arg.B[index.colB[k][i][0]*arg.problem_size.k()+index.rowB[k][i][0]],4,flag[0]*4);
                LDGSTSCA(&B[index.tileidx[k*2+i]+1],&arg.B[index.colB[k][i][1]*arg.problem_size.k()+index.rowB[k][i][1]],4,flag[1]*4);
                LDGSTSCA(&B[index.tileidx[k*2+i]+2],&arg.B[index.colB[k][i][2]*arg.problem_size.k()+index.rowB[k][i][2]],4,flag[2]*4);
                LDGSTSCA(&B[index.tileidx[k*2+i]+3],&arg.B[index.colB[k][i][3]*arg.problem_size.k()+index.rowB[k][i][3]],4,flag[3]*4);
            }
        }
    }

    for(int k=0;k<2;k++){
        for(int i=0;i<2;i++){
            for(int j=0;j<4;j++){
                index.rowB[k][i][j] += 16;
            }
        }
    }
}

__global__ void GEMM_MMA(MMAarguments arg){
    // __shared__ ElementInputA tileA[4][256*8];
    // __shared__ ElementInputB tileB[4][8*256];

    extern __shared__ ElementInputA tileA[];

    ElementInputB *tileB = reinterpret_cast<ElementInputB*>(&tileA[64*128]);

    ElementOutput C_fragment1[64],C_fragment2[64];
    ElementInputA A_fragment[32];
    ElementInputB B_fragment[32];

    struct Index index;

    const int iters = DIV(arg.problem_size.k(),16);
    
    loadtileC(arg,C_fragment1,C_fragment2,index);

    loadtileA(arg,&tileA[0],index);
    loadtileB(arg,&tileB[0],index);
    asm("cp.async.commit_group;\n"::);

    loadtileA(arg,&tileA[1*256*8],index);
    loadtileB(arg,&tileB[1*256*8],index);
    asm("cp.async.commit_group;\n"::);

    loadtileA(arg,&tileA[2*256*8],index);
    loadtileB(arg,&tileB[2*256*8],index);
    asm("cp.async.commit_group;\n"::);

    for(int i=0;i<iters;i++){
        asm("cp.async.wait_group 2;\n"::);
        __syncthreads();

        ldsA(&tileA[i%4*256*8],A_fragment,index);
        ldsB(&tileB[i%4*256*8],B_fragment,index);
        mma_tile(arg,A_fragment,B_fragment,C_fragment1,C_fragment2);
        
        loadtileA(arg,&tileA[(i+3)%4*256*8],index);
        loadtileB(arg,&tileB[(i+3)%4*256*8],index); 
        asm("cp.async.commit_group;\n"::); 
    }

    storetile(arg,C_fragment1,C_fragment2,index);
}

void launch_GEMM_MMA(MMAarguments &arg){
    dim3 grid,block;
    int smem_size;
    // threadblockShape 128 128 8
    // warpShape 64 64 8
    // every block has 4 warps

    grid.x = DIV(arg.problem_size.n(),128) * (1<<logtile);
    grid.y = DIV(DIV(arg.problem_size.m(),128),1<<logtile);
    grid.z = 1;

    block.x = 128;
    block.y = 1;
    block.z = 1;

    smem_size = 4*256*8*sizeof(ElementInputA)*2;

    CUDA_CHECK(cudaFuncSetAttribute((void *)GEMM_MMA, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    GEMM_MMA<<<grid,block,smem_size>>>(arg);
}

// Create a tuple of problem size for matrix multiplication
cutlass::gemm::GemmCoord problem_size;

// Initialize tensors using CUTLASS helper functions
cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a;  // <- Create matrix A with dimensions M x K
cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_b;  // <- Create matrix B with dimensions K x N
cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_c;  // <- Create matrix C with dimensions M x N
cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_d;  

GpuTimer timer;

int main(int argc,const char **argv){
    //////////////////////////INIT////////////////////////////////
    Options options;
    options.parse(argc, argv);

    if (options.help) {
        options.print_usage(std::cout) << std::endl;
        return 0;
    }

    problem_size = options.problem_size;

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
    },options.iterations);
    //////////////////////GEMM_MMA///////////////////////
    {   
        cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_mma_d(problem_size.mn());
        cutlass::reference::device::TensorFillRandomUniform(
            tensor_mma_d.device_view(),
            3,
            ElementAccumulator(0.f),
            ElementAccumulator(0.f),
            0
        );

        timer.bind_run("MMA_tune",[&]{
            MMAarguments mmaArg{
                problem_size,
                tensor_a.device_data(),
                tensor_b.device_data(),
                tensor_c.device_data(),
                tensor_mma_d.device_data()
            };
            launch_GEMM_MMA(mmaArg);
        },options.iterations);
        
        timer.testEqual<ElementOutput,LayoutOutput>("MMA_tune==ref",tensor_d,tensor_mma_d,options.ifprint);
        // timer.printTensor<ElementInputA,LayoutInputA>("A",tensor_a);
    }

}