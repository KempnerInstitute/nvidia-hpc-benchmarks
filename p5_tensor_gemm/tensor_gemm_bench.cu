/*
 * p5_tensor_gemm/tensor_gemm_bench.cu
 *
 * Compact cuBLASLt-based GEMM benchmark for tensor-core saturation testing.
 * Supports FP16 and FP8 (E4M3) on Hopper; FP4 gated to Blackwell (SM ≥ 100).
 *
 * Build (inside the NVIDIA HPC container or with CUDA ≥ 12.0):
 *   nvcc -O3 -std=c++17 -lcublasLt -lcublas -o tensor_gemm_bench tensor_gemm_bench.cu
 *
 * Usage:
 *   ./tensor_gemm_bench --dtype fp16 --m 8192 --n 8192 --k 8192 [--warmup 10] [--iters 100]
 *   ./tensor_gemm_bench --dtype fp8  --m 16384 --n 16384 --k 16384
 *
 * Output (per shape):
 *   dtype  M      N      K      time_ms   TFLOPS
 *   fp16   8192   8192   8192   0.347     3170.2
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <chrono>

#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cuda_fp16.h>

#if __CUDACC_VER_MAJOR__ >= 12
#include <cuda_fp8.h>
#endif

// ── Error checking ──────────────────────────────────────────────────────────
#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                   \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

#define CHECK_CUBLAS(call)                                                     \
    do {                                                                       \
        cublasStatus_t st = (call);                                            \
        if (st != CUBLAS_STATUS_SUCCESS) {                                     \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, \
                    (int)st);                                                  \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

// ── Dtype helpers ───────────────────────────────────────────────────────────
struct DtypeInfo {
    cudaDataType_t cuda_type;
    cublasComputeType_t compute_type;
    cudaDataType_t scale_type;   // type of alpha/beta
    size_t elem_bytes;
    const char* name;
    double flop_multiplier;      // 1.0 for fp16, 2.0 for fp8 (counts as 2× throughput class)
};

static DtypeInfo get_dtype_info(const std::string& dtype_str, int sm_version) {
    DtypeInfo info{};

    if (dtype_str == "fp16" || dtype_str == "FP16") {
        info.cuda_type    = CUDA_R_16F;
        info.compute_type = CUBLAS_COMPUTE_16F;
        info.scale_type   = CUDA_R_16F;
        info.elem_bytes   = 2;
        info.name         = "fp16";
        info.flop_multiplier = 1.0;
    }
#if __CUDACC_VER_MAJOR__ >= 12
    else if (dtype_str == "fp8" || dtype_str == "FP8" ||
             dtype_str == "fp8_e4m3" || dtype_str == "FP8_E4M3") {
        if (sm_version < 90) {
            fprintf(stderr, "ERROR: FP8 requires SM ≥ 90 (Hopper). Detected SM %d.\n", sm_version);
            exit(1);
        }
        info.cuda_type    = CUDA_R_8F_E4M3;
        info.compute_type = CUBLAS_COMPUTE_32F;
        info.scale_type   = CUDA_R_32F;
        info.elem_bytes   = 1;
        info.name         = "fp8_e4m3";
        info.flop_multiplier = 1.0;
    }
#endif
    else if (dtype_str == "fp4" || dtype_str == "FP4" ||
             dtype_str == "nvfp4" || dtype_str == "NVFP4") {
        if (sm_version < 100) {
            fprintf(stderr, "ERROR: FP4 requires SM ≥ 100 (Blackwell). Detected SM %d.\n", sm_version);
            fprintf(stderr, "  H100/H200 (Hopper, SM 90) do NOT support native FP4 tensor cores.\n");
            exit(1);
        }
        fprintf(stderr, "ERROR: FP4 GEMM support is not yet implemented.\n");
        fprintf(stderr, "  This is a placeholder for Blackwell-class GPUs.\n");
        exit(1);
    }
    else {
        fprintf(stderr, "ERROR: Unknown dtype '%s'. Supported: fp16, fp8, fp8_e4m3\n",
                dtype_str.c_str());
        exit(1);
    }

    return info;
}

// ── Get SM version ──────────────────────────────────────────────────────────
static int get_sm_version() {
    int device = 0;
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
    return prop.major * 10 + prop.minor;
}

static void print_gpu_info() {
    int device = 0;
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
    printf("GPU: %s (SM %d%d, %d SMs, %.1f GB, %d MHz)\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount,
           prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0),
           prop.clockRate / 1000);
}

// ── GEMM benchmark core ─────────────────────────────────────────────────────
static double benchmark_gemm(cublasLtHandle_t ltHandle,
                             int M, int N, int K,
                             const DtypeInfo& dt,
                             int warmup, int iters) {
    // Allocate device memory for A, B (input type)
    size_t sizeA = (size_t)M * K * dt.elem_bytes;
    size_t sizeB = (size_t)K * N * dt.elem_bytes;

    // Determine output type — FP8 inputs require FP16/BF16/FP32 output on Hopper
    cudaDataType_t c_type = dt.cuda_type;
    size_t c_elem_bytes = dt.elem_bytes;
#if __CUDACC_VER_MAJOR__ >= 12
    if (dt.cuda_type == CUDA_R_8F_E4M3) {
        c_type = CUDA_R_16F;      // accumulate into FP16
        c_elem_bytes = 2;
    }
#endif
    size_t sizeC = (size_t)M * N * c_elem_bytes;

    void *dA, *dB, *dC;
    CHECK_CUDA(cudaMalloc(&dA, sizeA));
    CHECK_CUDA(cudaMalloc(&dB, sizeB));
    CHECK_CUDA(cudaMalloc(&dC, sizeC));

    // Initialize with random-ish data (just fill with small values)
    CHECK_CUDA(cudaMemset(dA, 0x3C, sizeA));  // ~1.0 in fp16
    CHECK_CUDA(cudaMemset(dB, 0x3C, sizeB));
    CHECK_CUDA(cudaMemset(dC, 0x00, sizeC));

    // Create matrix descriptors
    cublasLtMatmulDesc_t operationDesc;
    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc;

    CHECK_CUBLAS(cublasLtMatmulDescCreate(&operationDesc, dt.compute_type, dt.scale_type));

    // FP8 on Hopper requires TN layout (TRANSA=T, TRANSB=N) — no NN kernel exists
    cublasOperation_t opA = CUBLAS_OP_N;
    cublasOperation_t opB = CUBLAS_OP_N;
#if __CUDACC_VER_MAJOR__ >= 12
    if (dt.cuda_type == CUDA_R_8F_E4M3) {
        opA = CUBLAS_OP_T;
    }
#endif
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA,
                                                 &opA, sizeof(opA)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB,
                                                 &opB, sizeof(opB)));

    // FP8 requires per-tensor scaling factors on device
    float *d_a_scale = nullptr, *d_b_scale = nullptr, *d_d_scale = nullptr;
#if __CUDACC_VER_MAJOR__ >= 12
    if (dt.cuda_type == CUDA_R_8F_E4M3) {
        float h_one = 1.0f;
        CHECK_CUDA(cudaMalloc(&d_a_scale, sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_b_scale, sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_d_scale, sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_a_scale, &h_one, sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_b_scale, &h_one, sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_d_scale, &h_one, sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc,
            CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_a_scale, sizeof(d_a_scale)));
        CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc,
            CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_b_scale, sizeof(d_b_scale)));
        CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc,
            CUBLASLT_MATMUL_DESC_D_SCALE_POINTER, &d_d_scale, sizeof(d_d_scale)));
    }
#endif

    // Matrix layout depends on transpose ops
    // NN: A(M,K,ldA=M), B(K,N,ldB=K)  => C(M,N)
    // TN: A(K,M,ldA=K), B(K,N,ldB=K)  => C(M,N)  (A stored as K×M)
    int ldA = (opA == CUBLAS_OP_T) ? K : M;
    int rowA = (opA == CUBLAS_OP_T) ? K : M;
    int colA = (opA == CUBLAS_OP_T) ? M : K;

    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Adesc, dt.cuda_type, rowA, colA, ldA));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Bdesc, dt.cuda_type, K, N, K));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Cdesc, c_type, M, N, M));

    // Alpha / Beta
    float alpha_f32 = 1.0f, beta_f32 = 0.0f;
    __half alpha_f16 = __float2half(1.0f), beta_f16 = __float2half(0.0f);

    void* alpha_ptr;
    void* beta_ptr;
    if (dt.scale_type == CUDA_R_16F) {
        alpha_ptr = &alpha_f16;
        beta_ptr  = &beta_f16;
    } else {
        alpha_ptr = &alpha_f32;
        beta_ptr  = &beta_f32;
    }

    // Heuristic to pick the best algo
    cublasLtMatmulPreference_t preference;
    CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&preference));
    size_t workspaceSize = 32 * 1024 * 1024;  // 32 MiB workspace
    CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspaceSize, sizeof(workspaceSize)));

    void* workspace;
    CHECK_CUDA(cudaMalloc(&workspace, workspaceSize));

    cublasLtMatmulHeuristicResult_t heuristicResult;
    int returnedAlgoCount = 0;
    cublasStatus_t algoStatus = cublasLtMatmulAlgoGetHeuristic(
        ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Cdesc,
        preference, 1, &heuristicResult, &returnedAlgoCount);

    if (algoStatus != CUBLAS_STATUS_SUCCESS || returnedAlgoCount == 0) {
        fprintf(stderr, "WARNING: No suitable algorithm found for %s M=%d N=%d K=%d. Skipping.\n",
                dt.name, M, N, K);
        cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(workspace);
        if (d_a_scale) cudaFree(d_a_scale);
        if (d_b_scale) cudaFree(d_b_scale);
        if (d_d_scale) cudaFree(d_d_scale);
        cublasLtMatmulDescDestroy(operationDesc);
        cublasLtMatrixLayoutDestroy(Adesc);
        cublasLtMatrixLayoutDestroy(Bdesc);
        cublasLtMatrixLayoutDestroy(Cdesc);
        cublasLtMatmulPreferenceDestroy(preference);
        return -1.0;
    }

    // Warmup
    for (int i = 0; i < warmup; i++) {
        CHECK_CUBLAS(cublasLtMatmul(ltHandle, operationDesc,
                                     alpha_ptr, dA, Adesc, dB, Bdesc,
                                     beta_ptr, dC, Cdesc, dC, Cdesc,
                                     &heuristicResult.algo,
                                     workspace, workspaceSize, 0));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Timed iterations
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; i++) {
        CHECK_CUBLAS(cublasLtMatmul(ltHandle, operationDesc,
                                     alpha_ptr, dA, Adesc, dB, Bdesc,
                                     beta_ptr, dC, Cdesc, dC, Cdesc,
                                     &heuristicResult.algo,
                                     workspace, workspaceSize, 0));
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));
    double avg_ms = (double)total_ms / iters;

    // Cleanup
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(workspace);
    if (d_a_scale) cudaFree(d_a_scale);
    if (d_b_scale) cudaFree(d_b_scale);
    if (d_d_scale) cudaFree(d_d_scale);
    cublasLtMatmulDescDestroy(operationDesc);
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Cdesc);
    cublasLtMatmulPreferenceDestroy(preference);

    return avg_ms;
}

// ── CLI parsing ─────────────────────────────────────────────────────────────
struct BenchConfig {
    std::string dtype = "fp16";
    std::vector<int> Ms, Ns, Ks;
    int warmup = 10;
    int iters  = 100;
};

static void print_usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s [options]\n"
        "\n"
        "Options:\n"
        "  --dtype <fp16|fp8|fp8_e4m3|fp4>   Data type (default: fp16)\n"
        "  --m <int[,int,...]>               M dimensions (default: 8192)\n"
        "  --n <int[,int,...]>               N dimensions (default: 8192)\n"
        "  --k <int[,int,...]>               K dimensions (default: 8192)\n"
        "  --warmup <int>                    Warmup iterations (default: 10)\n"
        "  --iters <int>                     Timed iterations (default: 100)\n"
        "  --help                            Show this message\n"
        "\n"
        "Examples:\n"
        "  %s --dtype fp16 --m 8192 --n 8192 --k 8192\n"
        "  %s --dtype fp8 --m 4096,8192,16384 --n 4096,8192,16384 --k 4096,8192,16384\n",
        prog, prog, prog);
}

static std::vector<int> parse_int_list(const char* str) {
    std::vector<int> vals;
    char* buf = strdup(str);
    char* tok = strtok(buf, ",");
    while (tok) {
        vals.push_back(atoi(tok));
        tok = strtok(nullptr, ",");
    }
    free(buf);
    return vals;
}

static BenchConfig parse_args(int argc, char** argv) {
    BenchConfig cfg;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--dtype") == 0 && i + 1 < argc) {
            cfg.dtype = argv[++i];
        } else if (strcmp(argv[i], "--m") == 0 && i + 1 < argc) {
            cfg.Ms = parse_int_list(argv[++i]);
        } else if (strcmp(argv[i], "--n") == 0 && i + 1 < argc) {
            cfg.Ns = parse_int_list(argv[++i]);
        } else if (strcmp(argv[i], "--k") == 0 && i + 1 < argc) {
            cfg.Ks = parse_int_list(argv[++i]);
        } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            cfg.warmup = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            cfg.iters = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            exit(0);
        } else {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            print_usage(argv[0]);
            exit(1);
        }
    }

    // Defaults
    if (cfg.Ms.empty()) cfg.Ms = {8192};
    if (cfg.Ns.empty()) cfg.Ns = {8192};
    if (cfg.Ks.empty()) cfg.Ks = {8192};

    return cfg;
}

// ── Main ────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    BenchConfig cfg = parse_args(argc, argv);

    int sm = get_sm_version();
    print_gpu_info();
    printf("\n");

    DtypeInfo dt = get_dtype_info(cfg.dtype, sm);

    cublasLtHandle_t ltHandle;
    CHECK_CUBLAS(cublasLtCreate(&ltHandle));

    // Print header
    printf("%-12s %8s %8s %8s %10s %10s\n",
           "dtype", "M", "N", "K", "time_ms", "TFLOPS");
    printf("──────────── ──────── ──────── ──────── ────────── ──────────\n");

    double best_tflops = 0.0;

    // Sweep all combinations of M, N, K
    for (int M : cfg.Ms) {
        for (int N : cfg.Ns) {
            for (int K : cfg.Ks) {
                double avg_ms = benchmark_gemm(ltHandle, M, N, K, dt,
                                                cfg.warmup, cfg.iters);
                if (avg_ms < 0) continue;  // algo not found

                double flops = 2.0 * (double)M * (double)N * (double)K;
                double tflops = (flops / (avg_ms * 1e-3)) / 1e12;

                printf("%-12s %8d %8d %8d %10.3f %10.1f\n",
                       dt.name, M, N, K, avg_ms, tflops);
                fflush(stdout);

                if (tflops > best_tflops) best_tflops = tflops;
            }
        }
    }

    printf("\n");
    printf("Best achieved: %.1f TFLOPS (%s)\n", best_tflops, dt.name);

    // Print theoretical peak context
    if (sm == 90) {
        // Detect GPU model for correct peak values
        // H100 SXM: 989.4 FP16, 1978.9 FP8 (1980 MHz boost)
        // H200 SXM: 989.4 FP16, 1978.9 FP8 (spec sheet same as H100 SXM)
        // Both H100/H200 share same SM90 TC peak specification
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        const char* gpu_label = "H100/H200 SXM";
        double peak_fp16 = 989.4;
        double peak_fp8  = 1978.9;
        if (strstr(prop.name, "H200")) {
            gpu_label = "H200 SXM";
        } else if (strstr(prop.name, "H100")) {
            gpu_label = "H100 SXM";
        }
        double peak = (cfg.dtype.find("fp8") != std::string::npos) ? peak_fp8 : peak_fp16;
        printf("%s theoretical peak (%s dense): %.1f TFLOPS\n", gpu_label, dt.name, peak);
        printf("Achieved %.1f%% of theoretical peak\n", 100.0 * best_tflops / peak);
    }

    CHECK_CUBLAS(cublasLtDestroy(ltHandle));
    return 0;
}
