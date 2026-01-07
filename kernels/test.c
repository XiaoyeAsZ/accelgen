#define N 200
#define M 300
#define K 400
#define DATA_TYPE float

void matmul(DATA_TYPE A[N][K], DATA_TYPE B[K][M], DATA_TYPE C[N][M]) {
  int i, j, k;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      for (int k = 0; k < K; k++) {
                C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
}

void conv2d(DATA_TYPE *act, DATA_TYPE *weight, DATA_TYPE *result,
            unsigned int n, unsigned int nic, unsigned int wi, unsigned int hi,
            unsigned int noc, unsigned int wk, unsigned int hk) {
  for (unsigned int i = 0; i < n; i++) {
    /* code */
  }

  return;
}