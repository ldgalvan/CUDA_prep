#include <cuda_runtime.h>
#include <stdio.h>
#include <fstream>

__global__ void matmul( float* A,  float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    if (row < n && col < n) {
        for (int k = 0; k < n; ++k)
            sum += A[row * n + k] * B[k * n + col];
        C[row * n + col] = sum;
    }
}

void benchmark_matmul(int N, std::ofstream& fout) {
    size_t size = N * N;
    size_t bytes = size * sizeof(float);

    float *A, *B, *C;
    cudaMallocManaged(&A, bytes);
    cudaMallocManaged(&B, bytes);
    cudaMallocManaged(&C, bytes);

    for (int i = 0; i < size; ++i) {
        A[i] = 1.0f;
        B[i] = 2.0f;
    }

    dim3 threads(32, 32);
    dim3 blocks((N + 31) / 32, (N + 31) / 32);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matmul<<<blocks, threads>>>(A, B, C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    double flops = 2.0 * N * N * N;
    double gflops = flops / (ms / 1000.0) / 1e9;
    double bytes_moved = 3.0 * size * sizeof(float);
    double bandwidth = bytes_moved / (ms / 1000.0) / 1e9;
    double intensity = flops / bytes_moved;

    printf("N=%d | Time=%.3f ms | GFLOPs=%.2f | Bandwidth=%.2f GB/s | Intensity=%.2f flop/byte\n",
           N, ms, gflops, bandwidth, intensity);

    fout << N << "," << ms << "," << gflops << "," << bandwidth << "," << intensity << "\n";

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
}

int main() {
    std::ofstream fout("roofline_results4.csv");
    fout << "matrix_size,time_ms,gflops,bandwidth_gbs,flop_per_byte\n";

    int sizes[] = {
        17, 33, 49, 65, 97, 129, 193, 257,
        513, 769, 1025, 1537, 2049, 4097, 8193, 16385, 32769
    };
        for (int N : sizes) {
        benchmark_matmul(N, fout);
    }

    fout.close();
    printf("Saved CSV to 'roofline_results4.csv'\n");
    return 0;
}
