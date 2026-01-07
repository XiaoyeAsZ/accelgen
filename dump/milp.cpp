#include <glpk.h>
#include <iostream>

int main() {
  glp_prob* lp = glp_create_prob();
  glp_set_prob_name(lp, "simple_milp");
  glp_set_obj_dir(lp, GLP_MAX);

  // -----------------------
  // Add constraints (rows)
  // -----------------------
  glp_add_rows(lp, 2);

  glp_set_row_name(lp, 1, "c1");
  glp_set_row_bnds(lp, 1, GLP_UP, 0.0, 4.0);  // 2x + y <= 4

  glp_set_row_name(lp, 2, "c2");
  glp_set_row_bnds(lp, 2, GLP_UP, 0.0, 5.0);  // x + 2y <= 5

  // -----------------------
  // Add variables (cols)
  // -----------------------
  glp_add_cols(lp, 2);

  // x (integer)
  glp_set_col_name(lp, 1, "x");
  glp_set_col_bnds(lp, 1, GLP_LO, 0.0, 0.0);  // x >= 0
  glp_set_col_kind(lp, 1, GLP_IV);            // integer
  glp_set_obj_coef(lp, 1, 3.0);

  // y (continuous)
  glp_set_col_name(lp, 2, "y");
  glp_set_col_bnds(lp, 2, GLP_LO, 0.0, 0.0);  // y >= 0
  glp_set_obj_coef(lp, 2, 2.0);

  // -----------------------
  // Constraint matrix
  // -----------------------
  // Matrix in COO format (1-based)
  int ia[5], ja[5];
  double ar[5];

  // Row 1: 2x + 1y
  ia[1] = 1;
  ja[1] = 1;
  ar[1] = 2.0;
  ia[2] = 1;
  ja[2] = 2;
  ar[2] = 1.0;

  // Row 2: 1x + 2y
  ia[3] = 2;
  ja[3] = 1;
  ar[3] = 1.0;
  ia[4] = 2;
  ja[4] = 2;
  ar[4] = 2.0;

  glp_load_matrix(lp, 4, ia, ja, ar);

  // -----------------------
  // Solve MILP
  // -----------------------
  glp_iocp parm;
  glp_init_iocp(&parm);
  parm.presolve = GLP_ON;

  int ret = glp_intopt(lp, &parm);

  if (ret == 0) {
    std::cout << "Optimal solution found\n";
    std::cout << "Objective = " << glp_mip_obj_val(lp) << "\n";
    std::cout << "x = " << glp_mip_col_val(lp, 1) << "\n";
    std::cout << "y = " << glp_mip_col_val(lp, 2) << "\n";
  } else {
    std::cout << "Solver failed\n";
  }

  glp_delete_prob(lp);
  return 0;
}