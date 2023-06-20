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
#include <limits>

namespace py = pybind11;//

const auto U = 1;
//const auto N = 1000;
const auto epsilon = 1;
const auto delta_u = 0.022;
//const auto alpha = 1 - 1/std::sqrt(N);



struct SupcritException {};


using namespace std;
using namespace arma;

using inds_t = unsigned short;
using avalanche_t = vector<inds_t>;

const inds_t STEP_SEP = numeric_limits<inds_t>::max();

struct EHE {

  vector<inds_t> avalanches;
  vector<unsigned int> avs_inds;
  Col<inds_t> avs;
  Col<unsigned int> inds;
  bool record_units = false;
  mat unit_hist;

  template <typename Func>
  void handle_avalanche(unsigned int start_unit,vec  &units,
                        Func &int_func, double epsilon=1, int U=U) {
    units(start_unit) = epsilon*(units(start_unit) - U);
    uvec A = {start_unit};
    vector<inds_t> Avec;
    unsigned int avalanche_duration = 1;
    bool warning_given=false;
    while(!A.is_empty()) {
      if(!warning_given && avalanche_duration > units.n_elem) {
        cout << "WARNING: Avalanche duration bigger than number of units!! " << avalanche_duration << endl;
        warning_given = true;
      } else if (avalanche_duration >= units.n_elem + 5) {
        cout << "TOO BIG I QUIT!" << endl;
        throw SupcritException();
      }
      int_func(units,A);
      units.elem(find(units < 0)).zeros(); // reset negative values to zero
      Avec = conv_to<vector<inds_t>>::from(move(A));
      avalanches.insert(avalanches.end(),make_move_iterator(Avec.begin()),make_move_iterator(Avec.end()));
      A = find(units > U);
      if(!A.is_empty()) avalanches.push_back(STEP_SEP);
      units.elem(A) = epsilon*(units.elem(A) - U);
      avalanche_duration++;
    }
  }

  template <typename Func>
  void simulate_model(vec &units,int numAvalanches,double deltaU, Func int_func) {
    avalanches = vector<inds_t>();
    avs_inds = vector<unsigned int>();
    avs_inds.reserve(numAvalanches);
    avalanches.reserve(10*numAvalanches); //reserve least amount of space needed
    vector<vec> uh;
    /*if(record_units) {
      unit_hist = mat(numAvalanches,units.n_elem);
    }*/
   // cout << "hi2" << endl;
    std::random_device rd;  
    std::mt19937 gen(rd()); 
    std::uniform_int_distribution<> dis(0, units.n_elem - 1);
    auto avalanche_counter=0;
    unsigned int r;
    while(avalanche_counter < numAvalanches){
      r = dis(gen);
      units(r) += deltaU;  
      if(units(r) >= U) {
        avs_inds.push_back(avalanches.size());
        try {
          handle_avalanche(r,units,int_func,1,U);
        } catch(SupcritException & ex) {
          cout << "catch exception end now" << endl;
          avalanches.push_back(STEP_SEP); //two following STEP_SEPs at the end signal error
          avalanches.push_back(STEP_SEP);
          break;
        }
        if(avalanche_counter % 1000 == 0){ 
          cout << "avalanche counter " << avalanche_counter << endl;
        }
        avalanche_counter++;
      }
      if(record_units) {
          //unit_hist.row(avalanche_counter) = units.t();
          uh.push_back(units);
      }
    }
    if(record_units) {
      unit_hist = mat(uh.size(),units.n_elem);
      for(unsigned int i = 0; i < uh.size();i++) {
        unit_hist.row(i) = uh[i].t();
      }
    }
    avs  = arma::Mat<inds_t>(avalanches.data(),avalanches.size(),1,false,true);
    inds = arma::Mat<unsigned int>(avs_inds.data(),avs_inds.size(),1,false,true);
  }


  tuple<vector<int>,vector<int>> get_avs_size_and_duration(Col<inds_t> &avs,vector<unsigned int> &avs_inds) {
    vector<int> avs_sizes;
    vector<int> avs_durations;
    Col<inds_t> avalanches;
    int step_separators;
    for(unsigned int i=0; i < avs_inds.size()-1;i++) {
      avalanches = avs.subvec(avs_inds[i],avs_inds[i+1]-1);
      step_separators = static_cast<uvec>(find(avalanches == STEP_SEP)).n_elem;
      avs_durations.push_back(step_separators+1);
      avs_sizes.push_back(avs_inds[i+1] - avs_inds[i] - step_separators);
    }
    return make_tuple(avs_sizes,avs_durations);
  }


  vector<vector<vector<inds_t>>> get_spiking_patterns() {
    return get_spiking_patterns(this->avs,this->avs_inds);
  }

  vector<vector<vector<inds_t>>> get_spiking_patterns(Col<inds_t> & avs, vector<unsigned int> &avs_inds) {
    vector<vector<vector<inds_t>>> spiking_patterns;
    spiking_patterns.reserve(avs_inds.size());
    for(unsigned int i=0; i < avs_inds.size()-1;i++) {
      vector<vector<inds_t>> sp;
      vector<inds_t> cur_step;
      for(unsigned int j=avs_inds[i];j<avs_inds[i+1];j++) {
        if(avs[j] != STEP_SEP) {
          cur_step.push_back(avs[j]);
        } else {
          sp.push_back(move(cur_step));
          cur_step.clear();
        }
      }
      sp.push_back(move(cur_step));
      cur_step.clear();
      spiking_patterns.push_back(move(sp));
      sp.clear();
    }
    return spiking_patterns;
  }

  tuple<Col<inds_t>,vector<unsigned int>> subnetwork_avalanches(Col<inds_t> &avs,vector<unsigned int> &avs_inds,set<inds_t> &subnet_inds) {
    vector<unsigned int> subavs_inds = {0};
    vector<inds_t> subavs;
    unsigned int subavs_idx = 0;
    inds_t cur;
    unsigned int avs_inds_idx = 0;
    for(unsigned int avs_idx=0; avs_idx < avs.n_elem; avs_idx++) {
      if(avs_idx == avs_inds[avs_inds_idx]) { // reached a new avalanche, add it to subavalanche indices
        avs_inds_idx++;
        if (subavs_inds.back() != subavs_idx) { // but only if at least one element in subnetwork spiked
          subavs_inds.push_back(subavs_idx);
        }
      }
      cur = avs[avs_idx];
      if(cur == STEP_SEP) {
        if(subavs_idx > 0 && subavs.back() != cur) { // add step sep but not if no subnetwork element spiked since last step sep
          subavs.push_back(cur);subavs_idx++;
        } else if(subavs_idx > subavs_inds.back()) {
          //current avalanche contained subnetwork elements but not in this step,
          //-> next subnetwork index starts new avalanche
          subavs_inds.push_back(subavs_idx);
        }
      } else if(subnet_inds.find(cur) != subnet_inds.end()) {
        subavs.push_back(cur);subavs_idx++;
      }
      //otherwise a unit spiked that is not in the subnet - will be ignored
   
    if(subavs_inds.back() >= subavs.size()) {
      subavs_inds.pop_back(); //remove trailing avs of size 0
    }
    return make_tuple(conv_to<Col<inds_t>>::from(subavs),subavs_inds);
  }

};
};

int main() {
  EHE e;
  int N = 10000;
  vec units = randu<vec>(N);
  mat W = (randn<mat>(N,N) + 0.8)/N;
  e.simulate_model(units,10000000,delta_u,
                   [&](vec &units,uvec &inds) {
                     vec A(size(units),arma::fill::zeros);
                     A.elem(inds) += 1;
                     units += W*A;
                   });
  cout << "Am I sill alive?" << endl;
};



