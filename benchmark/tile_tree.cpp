extern int gelu(int x);

void krnl_gemm_gelu(int a[1024][4096], int b[4096][4096], int c[4096][4096],
                    int d[4096][4096]) {
  for (unsigned int i0 = 0; i0 < 1024; i0 += 64) {
    for (unsigned int j0 = 0; j0 < 4096; j0 += 64) {
      for (unsigned int k0 = 0; k0 < 4096; k0 += 64) {
        for (unsigned int k1 = 0; k1 < 64; k1 += 1) {
          for (unsigned int i1 = 0; i1 < 64; i1 += 1) {
            for (unsigned int j1 = 0; j1 < 64; j1 += 1) {
              c[i0 + i1][j0 + j1] += a[i0 + i1][k0 + k1] * b[k0 + k1][j0 + j1];
            }
          }
        }
      }

      for (unsigned int i1 = 0; i1 < 64; i1 += 1) {
        for (unsigned int j1 = 0; j1 < 64; j1 += 1) {
          d[i0 + i1][j0 + j1] += gelu(d[i0 + i1][j0 + j1]);
        }
      }
    }
  }
}