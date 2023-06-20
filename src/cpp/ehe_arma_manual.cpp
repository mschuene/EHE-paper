/* <%
   setup_pybind11(cfg)
   cfg['compiler_args'] = ['-O3','-std=c++14','-march=native','-L/opt/openblas/lib/', '-lopenblas','-larmadillo']
   cfg['libraries'] = ['-L/opt/openblas/lib/', 'openblas']
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

namespace py = pybind11;

const auto U = 1;
const auto N = 1000;
const auto epsilon = 1;
const auto delta_u = 0.022;
const auto alpha = 1 - 1/std::sqrt(N);

using namespace std;
using namespace arma;

       std::tuple<int,int> handle_avalanche(int start_unit,vec & units,const mat & W,double epsilon=1, int U=U) {
       int avalanche_size = 0;
       int avalanche_duration = 0;
       vec A(size(units),fill::zeros);
       A(start_unit) = 1;
       units(start_unit) = epsilon*(units(start_unit) - U);
       int s = 1;
        // uvec inds;
        // inds << start_unit;
       while(s > 0) {
         avalanche_size += s;
         avalanche_duration += 1;
         //   cout << "before mult " << endl;
         //cout << "A before mult " << endl << A << endl;
         units += W.at(1,1) * A;
         s = 0;
         /*double u;
         for(auto i = 0u; i < units.n_elem; i++) {
           u = units[i];
           if (u > U) {
               units[i] = epsilon*(u - U);
               A[i] = 1;
               s++;
           } else {
             A[i] = 0;
           }
           }*/
         //cout << " after mult " << endl;

         // cout << "A after mult " << endl << A << endl;
         // cout << " inds " << endl << inds << endl;
         // s = 0;
         auto i = 0u;
         asm("#llambdas");
         units = units.for_each([&](vec::elem_type& u) {if(u > U) { A[i] = 1;s++;u = epsilon*(u - U);} else { A[i] = 0;}; i++;});
         asm("#llambdends");
         // A.zeros();
         // inds = find(units > U);
         // s = inds.n_elem;
         // A.elem(find(units > U)).ones();
         // cout << "units before resetting " << endl << units << endl; 
         //units.elem(inds) = epsilon*(units.elem(inds) - U);
//         cout << "before resetting " << endl;
          // A.zeros();
          // 
         //  cout << "after resetting " << endl;
//          cout << "i after resetting " << i << endl;
      //    cout << "new threshold crossings " << s << endl;
         // cout << "after resetting " << endl << "units " << endl << units << endl << "A " << endl << A << endl;
         
         }
       //cout << "avalanche size " << avalanche_size << " avalanche duration " << avalanche_duration << endl;
       return make_tuple(avalanche_size,avalanche_duration);
     }

auto simulate_model(vec & units,int numAvalanches,
                    const mat & W,double deltaU) {
  // std::cout << "original units is " << std::endl << units << std::endl
  //           << "original W is " << std::endl << W << std::endl;
  std::list<int> avalanche_sizes {};
  std::list<int> avalanche_durations {};
  auto avalanche_counter=0;
  std::random_device rd;  
  std::mt19937 gen(rd()); 
  std::uniform_int_distribution<> dis(1, units.n_elem - 1);
  int r;
  auto avd = 0;
  while(avd <= numAvalanches){
    r = dis(gen);
    units(r) += deltaU;
    if(units(r) >= U) {
//      cout << "new avalanche " << endl;
      auto av_tuple = handle_avalanche(r,units,W,1,U);
      avalanche_counter++;
      avalanche_sizes.push_back(std::get<0>(av_tuple));
      avalanche_durations.push_back(std::get<1>(av_tuple));
      avd += get<1>(av_tuple);
    }
  }
  return std::make_tuple(avalanche_sizes,avalanche_durations);
}


mat test(mat & test) {
  std::cout << "Hi there from c++" << std::endl;
  return test*2;
 }
/*
PYBIND11_PLUGIN(ehe_arma__module_suffix__) {

  py::module m("ehe_eigen", "Simulate ehe model with eigen code");

  m.def("simulate_model",&simulate_model,"simulate model in eigen");

  m.def("test",&test,"test");

  return m.ptr();

}
*/


int main(void) {
  int N = 10000;
  mat W = ones<mat>(N,N) * (1 - 1/sqrt(N)) / N;
  vec units = ones<vec>(N)/3;
  simulate_model(units,10000,W,0.022);
  return 1;            
  
}
