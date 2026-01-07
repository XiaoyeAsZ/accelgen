void ffn_vit(int x[256][128], int w1[128][256], int w2[256][128]) {}

void MM1(int x[256][128], int w1[128][256], int out1[256][256]) {

  unsigned int ti0, tj0, tk0;
  unsigned int ui, uj, uk;

  for (unsigned int i0 = 0; i0 < 256; i0 += ti0) {
    for (unsigned int j0 = 0; j0 < 256; j0 += tj0) {
      for (unsigned int k0 = 0; k0 < 128; k0 += tk0) {

        // Unroll
        for (unsigned int i1 = 0; i1 < ti0; i1 += ui) {
          // Unroll
          for (unsigned int j1 = 0; j1 < tj0; j1 += uj) {
            // Unroll Factor = 1
            for (unsigned int k1 = 0; k1 < tk0; k1 += uk) {
              out1[i0 + i1][j0 + j1] +=
                  x[i0 + i1][k0 + k1] * w1[k0 + k1][j0 + j1];
            }
          }
        }

      }
    }
  }


}

void MM1(int x[256][256], int w1[256][128], int out1[256][128]) {

  unsigned int ti0, tj0, tk0;
  unsigned int ui, uj, uk;

  for (unsigned int i0 = 0; i0 < 256; i0 += ti0) {
    for (unsigned int j0 = 0; j0 < 128; j0 += tj0) {
      for (unsigned int k0 = 0; k0 < 256; k0 += tk0) {

        // Unroll
        for (unsigned int i1 = 0; i1 < ti0; i1 += ui) {
          // Unroll
          for (unsigned int j1 = 0; j1 < tj0; j1 += uj) {
            // Unroll Factor = 1
            for (unsigned int k1 = 0; k1 < tk0; k1 += uk) {
              out1[i0 + i1][j0 + j1] +=
                  x[i0 + i1][k0 + k1] * w1[k0 + k1][j0 + j1];
            }
          }
        }

      }
    }
  }

  
}
