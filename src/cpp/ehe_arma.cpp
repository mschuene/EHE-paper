/* <%

   cfg['linker_args'] = ['-Wl,-rpath=/opt/openblas/lib -lopenblas']
   setup_pybind11(cfg)
   cfg['compiler_args'] = ['-O3','-std=c++11','-march=native']
   %>
*/
#define ARMA_DONT_USE_WRAPPER
#define ARMA_DONT_USE_LAPACK
#define ARMA_USE_BLAS

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <armadillo>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "pybind11_armadillo.h"
#include "arma_supportlib.h"

#include <cstdlib>
#include <cmath>
#include <tuple>
#include <list>
#include <random>
#include <iostream>

namespace py = pybind11;//

const auto U = 1;
const auto N = 1000;
const auto epsilon = 1;
const auto delta_u = 0.022;
const auto alpha = 1 - 1/std::sqrt(N);

using namespace std;
using namespace arma;

  template <typename Func>
  std::tuple<int,int> handle_avalanche(int start_unit,vec & units, Func& int_func,double epsilon=1, int U=U) {
  int avalanche_size = 0;
  int avalanche_duration = 0;
  vec A(size(units),fill::zeros);
  A(start_unit) = 1;
  units(start_unit) = epsilon*(units(start_unit) - U);
  int s = 1;
  while(s > 0) {
    avalanche_size += s;
    avalanche_duration += 1;
    int_func(units,A);//units += W * A;
    s = 0;
    auto i = 0u;
    units = units.for_each([&](vec::elem_type& u) {if(u > U) { A[i] = 1;s++;u = epsilon*(u - U);} else { A[i] = 0;}; i++;});
    }

  return make_tuple(avalanche_size,avalanche_duration);
}



template <typename Func>
tuple<vector<int>,vector<int>> simulate_model(vec & units,int numAvalanches,
                    Func int_func,double deltaU) {
  std::vector<int> avalanche_sizes(numAvalanches);
  std::vector<int> avalanche_durations(numAvalanches);
  auto avalanche_counter=0;
  std::random_device rd;  
  std::mt19937 gen(rd()); 
  std::uniform_int_distribution<> dis(0, units.n_elem - 1);
  int r;
  while(avalanche_counter < numAvalanches){
    r = dis(gen);
    units(r) += deltaU;
    if(units(r) >= U) {
      tie(avalanche_sizes[avalanche_counter],
          avalanche_durations[avalanche_counter]) = handle_avalanche(r,units,int_func,1,U);
      avalanche_counter++;
      if(avalanche_counter % 10000 == 0) { cout << "avalanche counter " << avalanche_counter << endl;}
    }
  }
  return std::make_tuple(avalanche_sizes,avalanche_durations);
}


mat test(mat & test) {
  std::cout << "Hi there from c++" << std::endl;
  return test*3;
 }

PYBIND11_PLUGIN(ehe_arma__module_suffix__) {

  py::module m("ehe_arma", "Simulate ehe model with arma code");

  m.def("simulate_model_mat",
        [](vec& units, int numAvalanches,const mat & W,double deltaU) 
          {py::gil_scoped_release release;
            return simulate_model(units,numAvalanches,[&](vec& units,vec& A) {units += W*A;},deltaU);},"doc");
  m.def("simulate_model_const",
        [](vec& units, int numAvalanches,double w,double deltaU)
          {py::gil_scoped_release release;
            return simulate_model(units,numAvalanches,[=](vec& units,vec& A) { units += w*sum(A);},deltaU);},"doc");

  m.def("test",&test,"test");

  return m.ptr();

}
