#include <omp.h>
#include <iostream>

int main() {
  // Force OpenMP not to reduce thread count automatically
  omp_set_dynamic(0);

  std::cout << "omp_get_max_threads() = " << omp_get_max_threads() << std::endl;

#pragma omp parallel
  {
#pragma omp single
    {
      std::cout << "Actual threads running = " << omp_get_num_threads()
                << std::endl;
    }

    std::cout << "Hello from thread " << omp_get_thread_num() << std::endl;
  }

  return 0;
}