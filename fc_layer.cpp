// fc_layer.cpp

// ./nnfc data/inputs.32.bin data/outputs.32.bin 4
// sudo perf stat -e cycles,instructions,branches,branch-misses,cache-references,cache-misses  ./nnfc data/inputs.32.bin data/outputs.32.bin 4


// ./nnfc data/inputs.256.bin data/outputs.256.bin 4
// sudo perf stat -e cycles,instructions,branches,branch-misses,cache-references,cache-misses  ./nnfc data/inputs.256.bin data/outputs.256.bin 4


// ./nnfc data/inputs.1024.bin data/outputs.1024.bin 4
// sudo perf stat -e cycles,instructions,branches,branch-misses,cache-references,cache-misses  ./nnfc data/inputs.1024.bin data/outputs.1024.bin 4



// ./nnfc data/inputs.4096.bin data/outputs.4096.bin 4
// sudo perf stat -e cycles,instructions,branches,branch-misses,cache-references,cache-misses ./nnfc data/inputs.4096.bin data/outputs.4096.bin 4



#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <thread>
#include <vector>
#include <algorithm>

static const size_t BLOCK_A = 16;  // input block
static const size_t BLOCK_O = 128;  // output block
static const size_t CHUNK = 32;    // input row chunk

static inline bool is_aligned_32(const void* p) {
    return (((uintptr_t)p) & 31) == 0;
}

static inline void accumulate_block(const float* __restrict mat,
                                    const float* __restrict in_row,
                                    float* __restrict out,
                                    size_t a_len, size_t o_len, size_t output_dim) {
    bool out_aligned = is_aligned_32(out);
    bool mat_aligned = is_aligned_32(mat);

    for (size_t ai = 0; ai < a_len; ++ai) {
        __m256 vinv = _mm256_set1_ps(in_row[ai]);
        const float* matrow = mat + ai * output_dim;
        size_t j = 0;

        for (; j + 16 <= o_len; j += 16) {
            __m256 mout1 = out_aligned ? _mm256_load_ps(out + j) : _mm256_loadu_ps(out + j);
            __m256 mout2 = out_aligned ? _mm256_load_ps(out + j + 8) : _mm256_loadu_ps(out + j + 8);
            __m256 mmat1 = mat_aligned ? _mm256_load_ps(matrow + j) : _mm256_loadu_ps(matrow + j);
            __m256 mmat2 = mat_aligned ? _mm256_load_ps(matrow + j + 8) : _mm256_loadu_ps(matrow + j + 8);

            
            mout1 = _mm256_fmadd_ps(vinv, mmat1, mout1);
            mout2 = _mm256_fmadd_ps(vinv, mmat2, mout2);

            if (out_aligned) {
                _mm256_store_ps(out + j, mout1);
                _mm256_store_ps(out + j + 8, mout2);
            } else {
                _mm256_storeu_ps(out + j, mout1);
                _mm256_storeu_ps(out + j + 8, mout2);
            }
        }

        for (; j < o_len; ++j)
            out[j] += in_row[ai] * matrow[j];
    }
}

static inline void add_dot_blocked(const float* __restrict matrix,
                                   const float* __restrict input_row,
                                   float* __restrict out,
                                   size_t input_dim, size_t output_dim) {
    for (size_t ao = 0; ao < input_dim; ao += BLOCK_A) {
        size_t a_len = std::min(BLOCK_A, input_dim - ao);
        for (size_t oo = 0; oo < output_dim; oo += BLOCK_O) {
            size_t o_len = std::min(BLOCK_O, output_dim - oo);
            accumulate_block(matrix + ao * output_dim + oo,
                             input_row + ao,
                             out + oo,
                             a_len, o_len, output_dim);
        }
    }
}

void fc_layer(size_t data_cnt, size_t input_dim, size_t output_dim,
              float* __restrict matrix, float* __restrict bias,
              float* __restrict input, float* __restrict output,
              int threads) {

    int nth = std::min(std::max(1, threads), (int)data_cnt);

    auto worker = [&](size_t start_idx, size_t end_idx) {
        for (size_t base = start_idx; base < end_idx; base += CHUNK) {
            size_t chunk_end = std::min(base + CHUNK, end_idx);
            for (size_t iidx = base; iidx < chunk_end; ++iidx) {
                const float* in_row = input + input_dim * iidx;
                float* out_row = output + output_dim * iidx;

               
                memcpy(out_row, bias, output_dim * sizeof(float));

                
                add_dot_blocked(matrix, in_row, out_row, input_dim, output_dim);

                // ReLU
                __m256 vzero = _mm256_setzero_ps();
                size_t j = 0;
                if (is_aligned_32(out_row)) {
                    for (; j + 8 <= output_dim; j += 8) {
                        __m256 v = _mm256_load_ps(out_row + j);
                        _mm256_store_ps(out_row + j, _mm256_max_ps(v, vzero));
                    }
                } else {
                    for (; j + 8 <= output_dim; j += 8) {
                        __m256 v = _mm256_loadu_ps(out_row + j);
                        _mm256_storeu_ps(out_row + j, _mm256_max_ps(v, vzero));
                    }
                }
                for (; j < output_dim; ++j)
                    if (out_row[j] < 0.0f) out_row[j] = 0.0f;
            }
        }
    };

    if (nth == 1) {
        worker(0, data_cnt);
    } else {
        std::vector<std::thread> th;
        th.reserve(nth);
        size_t base = 0;
        size_t bsize = data_cnt / nth;
        size_t rem = data_cnt % nth;

        for (int t = 0; t < nth; ++t) {
            size_t s = base;
            size_t e = s + bsize + (t < (int)rem ? 1 : 0);
            th.emplace_back([=]() { worker(s, e); });
            base = e;
        }
        for (auto &tt : th) tt.join();
    }
}
