// #include <random>
// #include <cmath>
#include "headers.h"

#define DATA_TYPE float

#define B 1
#define H 1
#define M 4096
#define N 4096
#define D 128

#define N_TILE 64
#define M_TILE 64

extern "C" void flash_attention(
    // Tensor defination
    DATA_TYPE query[B][H][M][D], DATA_TYPE key[B][H][N][D],
    DATA_TYPE value[B][H][N][D], DATA_TYPE mma1_result[B][H][M][N],
    DATA_TYPE exp_result[B][H][M][N], DATA_TYPE max_result[B][H][M],
    DATA_TYPE max_result_record[B][H][M], DATA_TYPE max_result_diff[B][H][M],
    DATA_TYPE max_result_diff_exp[B][H][M],
    DATA_TYPE exp_sum_result_record_revise[B][H][M],
    DATA_TYPE exp_sum_result[B][H][M], DATA_TYPE exp_sum_result_record[B][H][M],
    DATA_TYPE attn_result_revise0[B][H][M],
    DATA_TYPE attn_result_revise1[B][H][M],
    DATA_TYPE attn_result_revise[B][H][M][D],
    DATA_TYPE exp_sum_result_revise[B][H][M],
    DATA_TYPE softmax_result[B][H][M][N], DATA_TYPE mma2_result[B][H][M][D],
    DATA_TYPE attn_result[B][H][M][D]) {
  for (unsigned int ib = 0; ib < B; ib++) {
    for (unsigned int ih = 0; ih < H; ih++) {
      for (unsigned int im_t = 0; im_t < (M + M_TILE - 1) / M_TILE; im_t++) {
        for (unsigned int in_t = 0; in_t < (N + N_TILE - 1) / N_TILE; in_t++) {
          // Compute MMA in size of M_TILE x N_TILE
          for (unsigned int id = 0; id < D; id++) {
            for (unsigned int im = 0; im < M_TILE; im++) {
              for (unsigned int in = 0; in < N_TILE; in++) {
                mma1_result[ib][ih][im_t * M_TILE + im][in_t * N_TILE + in] +=
                    query[ib][ih][im_t * M_TILE + im][id] *
                    key[ib][ih][in_t * N_TILE + in][id];
              }
            }
          }

          // Compute max
          for (unsigned int im = 0; im < M_TILE; im++) {
            DATA_TYPE max_value = -0x7ffff;
            for (unsigned int in = 0; in < N_TILE; in++) {
              max_value = f32_max(
                  max_value,
                  mma1_result[ib][ih][im_t * M_TILE + im][in_t * N_TILE + in]);
            }
            max_result[ib][ih][im_t * M_TILE + im] = max_value;
          }

          // Update maximum
          for (unsigned int im = 0; im < M_TILE; im++) {
            max_result[ib][ih][im_t * M_TILE + im] =
                f32_max(max_result_record[ib][ih][im_t * M_TILE + im],
                        max_result[ib][ih][im_t * M_TILE + im]);
          }

          // Compute exp (safe version)
          for (unsigned int im = 0; im < M_TILE; im++) {
            for (unsigned int in = 0; in < N_TILE; in++) {
              exp_result[ib][ih][im_t * M_TILE + im][in_t * N_TILE + in] =
                  f32_exp(mma1_result[ib][ih][im_t * M_TILE + im]
                                     [in_t * N_TILE + in] -
                          max_result[ib][ih][im_t * M_TILE + im]);
            }
          }

          // Compute sum of exp
          for (unsigned int im = 0; im < M_TILE; im++) {
            for (unsigned int in = 0; in < N_TILE; in++) {
              exp_sum_result[ib][ih][im_t * M_TILE + im] +=
                  exp_result[ib][ih][im_t * M_TILE + im][in_t * N_TILE + in];
            }
          }

          // Cmpute difference between max(now) and max(recorded)
          for (unsigned int im = 0; im < M_TILE; im++) {
            for (unsigned int in = 0; in < N_TILE; in++) {
              max_result_diff[ib][ih][im_t * M_TILE + im] =
                  max_result_record[ib][ih][im_t * M_TILE + im] -
                  max_result[ib][ih][im_t * M_TILE + im];
            }
          }

          // Compute exp of max-diff
          for (unsigned int im = 0; im < M_TILE; im++) {
            for (unsigned int in = 0; in < N_TILE; in++) {
              max_result_diff_exp[ib][ih][im_t * M_TILE + im] =
                  f32_exp(max_result_diff[ib][ih][im_t * M_TILE + im]);
            }
          }

          // Cmpute revision of exp_sum
          for (unsigned int im = 0; im < M_TILE; im++) {
            for (unsigned int in = 0; in < N_TILE; in++) {
              exp_sum_result_record_revise[ib][ih][im_t * M_TILE + im] =
                  exp_sum_result_record[ib][ih][im_t * M_TILE + im] *
                  max_result_diff_exp[ib][ih][im_t * M_TILE + im];
            }
          }

          for (unsigned int im = 0; im < M_TILE; im++) {
            for (unsigned int in = 0; in < N_TILE; in++) {
              exp_sum_result_revise[ib][ih][im_t * M_TILE + im] =
                  exp_sum_result[ib][ih][im_t * M_TILE + im] +
                  exp_sum_result_record_revise[ib][ih][im_t * M_TILE + im];
            }
          }

          // Compute softmax
          for (unsigned int im = 0; im < M_TILE; im++) {
            for (unsigned int in = 0; in < N_TILE; in++) {
              softmax_result[ib][ih][im_t * M_TILE + im][in_t * N_TILE + in] =
                  exp_result[ib][ih][im_t * M_TILE + im][in_t * N_TILE + in] /
                  exp_sum_result_revise[ib][ih][im_t * M_TILE + im];
            }
          }

          for (unsigned int in = 0; in < N_TILE; in++) {
            for (unsigned int im = 0; im < M_TILE; im++) {
              for (unsigned int id = 0; id < D; id++) {
                mma2_result[ib][ih][im_t * M_TILE + im][id] +=
                    softmax_result[ib][ih][im_t * M_TILE + im]
                                  [in_t * N_TILE + in] *
                    value[ib][ih][in_t * N_TILE + in][id];
              }
            }
          }

          for (unsigned int im = 0; im < M_TILE; im++) {
            for (unsigned int in = 0; in < N_TILE; in++) {
              attn_result_revise0[ib][ih][im_t * M_TILE + im] =
                  exp_sum_result_record[ib][ih][im_t * M_TILE + im] /
                  exp_sum_result_revise[ib][ih][im_t * M_TILE + im];
            }
          }

          for (unsigned int im = 0; im < M_TILE; im++) {
            for (unsigned int in = 0; in < N_TILE; in++) {
              attn_result_revise1[ib][ih][im_t * M_TILE + im] =
                  max_result_diff_exp[ib][ih][im_t * M_TILE + im] *
                  attn_result_revise0[ib][ih][im_t * M_TILE + im];
            }
          }

          for (unsigned int im = 0; im < M_TILE; im++) {
            for (unsigned int id = 0; id < D; id++) {
              attn_result_revise[ib][ih][im_t * M_TILE + im][id] =
                  attn_result[ib][ih][im_t * M_TILE + im][id] *
                  attn_result_revise1[ib][ih][im_t * M_TILE + im];
            }
          }

          for (unsigned int im = 0; im < M_TILE; im++) {
            for (unsigned int id = 0; id < D; id++) {
              attn_result[ib][ih][im_t * M_TILE + im][id] =
                  attn_result_revise[ib][ih][im_t * M_TILE + im][id] +
                  mma2_result[ib][ih][im_t * M_TILE + im][id];
            }
          }
        }
      }
    }
  }
}

// void matmul_qk_ref(DATA_TYPE query[B][H][M][D], DATA_TYPE key[B][H][N][D],
//                    DATA_TYPE result[B][H][M][N]) {
//   for (unsigned int b = 0; b < B; b++) {
//     for (unsigned int h = 0; h < H; h++) {
//       for (unsigned int m = 0; m < M; m++) {
//         for (unsigned int n = 0; n < N; n++) {
//           for (unsigned int d = 0; d < D; d++) {
//             result[b][h][m][n] += query[b][h][m][d] * key[b][h][n][d];
//           }
//         }
//       }
//     }
//   }
// }

// void initialize_q(DATA_TYPE query[B][H][M][D]) {
//   std::random_device rd;
//   std::mt19937 gen(rd());
//   std::uniform_real_distribution<DATA_TYPE> dist(-1.0f, 1.0f);
//   for (unsigned int b = 0; b < B; b++) {
//     for (unsigned int h = 0; h < H; h++) {
//       for (unsigned int m = 0; m < M; m++) {
//         for (unsigned int d = 0; d < D; d++) {
//           query[b][h][m][d] = dist(gen);
//         }
//       }
//     }
//   }
// }

// void initialize_k(DATA_TYPE key[B][H][N][D]) {
//   std::random_device rd;
//   std::mt19937 gen(rd());
//   std::uniform_real_distribution<DATA_TYPE> dist(-1.0f, 1.0f);
//   for (unsigned int b = 0; b < B; b++) {
//     for (unsigned int h = 0; h < H; h++) {
//       for (unsigned int n = 0; n < N; n++) {
//         for (unsigned int d = 0; d < D; d++) {
//           key[b][h][n][d] = dist(gen);
//         }
//       }
//     }
//   }
// }

// DATA_TYPE query_test[B][H][M][D];
// DATA_TYPE key_test[B][H][N][D];
// DATA_TYPE result[B][H][M][N];
// DATA_TYPE result_ref[B][H][M][N];

// int main() {
//   initialize_q(query_test);
//   initialize_k(key_test);

//   matmul_qk(query_test, key_test, result);
//   matmul_qk_ref(query_test, key_test, result_ref);

//   for (unsigned int b = 0; b < B; b++) {
//     for (unsigned int h = 0; h < H; h++) {
//       for (unsigned int m = 0; m < M; m++) {
//         for (unsigned int n = 0; n < N; n++) {
//           if (result[b][h][m][n] != result_ref[b][h][m][n]) printf("err");
//         }
//       }
//     }
//   }

//   return 0;
// }