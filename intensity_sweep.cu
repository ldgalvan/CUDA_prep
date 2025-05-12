#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>


//gpu code
template<int FLOP_COUNT>
__global__ void vec_opN( float* A,  float* B,  float* D,  float* E, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float acc = 0.0f;
        for (int j = 0; j < FLOP_COUNT / 4; ++j) {
            acc += A[i] * B[i];
            acc += D[i] * E[i];
        }
        C[i] = acc;
    }
}

//host
template<int FLOP_COUNT>
void benchmark(std::ofstream& fout, int N) {
    size_t bytes = N * sizeof(float);
    float *A, *B, *C, *D, *E;
    cudaMallocManaged(&A, bytes);
    cudaMallocManaged(&B, bytes);
    cudaMallocManaged(&C, bytes);
    cudaMallocManaged(&D, bytes);
    cudaMallocManaged(&E, bytes);

    for (int i = 0; i < N; ++i) {
        A[i] = 1.0f;
        B[i] = 2.0f;
        D[i] = 3.0f;
        E[i] = 4.0f;
    }

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    vec_opN<FLOP_COUNT><<<blocks, threads>>>(A, B, D, E, C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    double flops = FLOP_COUNT * N;
    double bytes_moved = 5 * N * sizeof(float); 
    double gflops = flops / (ms / 1000.0) / 1e9;
    double bandwidth = bytes_moved / (ms / 1000.0) / 1e9;
    double intensity = flops / bytes_moved;

    fout << "vec_op" << FLOP_COUNT
         << "," << N << "," << ms << "," << gflops << "," << bandwidth << "," << intensity << "\n";

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    cudaFree(D);
    cudaFree(E);
}

int main() {
    const int vec_size = 1 << 16;
    std::ofstream fout("vec_op_byte_sweep33.csv");
    fout << "kernel,N,time_ms,gflops,bandwidth_gbs,intensity\n";

    std::vector<int> flops = {
        1,2,4,8,16,32,64,128, 256, 512, 768, 1024, 2048, 4096, 8192,
        16384, 32768, 65536, 131072, 262144,
        524288, 1048576, 1500000, 1750000, 2000000, 3000000, 4000000, 5000000, 6000000, 7000000
    };

    for (int f : flops) {
        switch (f) {
            #define RUN(F) case F: benchmark<F>(fout, vec_size); break;
            RUN(1) RUN(2) RUN(4) RUN(8) RUN(16) RUN(32) RUN(64) RUN(128)
            RUN(256) RUN(512) RUN(768) RUN(1024) RUN(2048) RUN(4096)
            RUN(8192) RUN(16384) RUN(32768) RUN(65536) RUN(131072)
            RUN(262144) RUN(524288) RUN(1048576) RUN(1500000)
            RUN(1750000) RUN(2000000) RUN(3000000) RUN(4000000) RUN(5000000) RUN(6000000) RUN(7000000)
        }
    }

    fout.close();
    std::cout << "Saved CSV to vec_op_byte_sweep33.csv\n";
    return 0;
}
