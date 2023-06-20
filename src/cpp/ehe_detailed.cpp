
/* <%

   cfg['linker_args'] = ['-lopenblas']
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


namespace py = pybind11;
using namespace std;
using namespace arma;
using inds_t = unsigned short;
using avalanche_t = vector<inds_t>;
const auto U = 1;
//const auto N = 1000;
//const auto epsilon = 1;
const auto delta_u = 0.022;
//const auto alpha = 1 - 1/std::sqrt(N);
const inds_t STEP_SEP = numeric_limits<inds_t>::max();

struct SupcritException {};

struct EHE__module_suffix__ {

  vector<inds_t> activated_units;
  vector<unsigned int> act_inds;
  vector<inds_t> avalanches;
  vector<unsigned int> avs_inds;
  Col<inds_t> avs;
  Col<unsigned int> inds;
  bool record_units = false;
  mat unit_hist;
  vec old_units;
  bool use_old_units = false;
  bool clamp_neg = false;
  vector<unsigned int> avs_sizes;
  vector<unsigned int> avs_durations;
  float max_duration_factor = 10;
  double tau = 1;  
  double epsilon = 1;
  bool random=false;

  template <typename Func>
  void handle_avalanche(vec  &units,
                        Func &int_func, double epsilon,int U,
                        vector<inds_t> &avalanches,
                        vector<unsigned int> &avs_sizes,
                        vector<unsigned int> &avs_durations) {
    vector<inds_t> Avec;
    unsigned int avalanche_duration = 0;
    unsigned int avalanche_size = 0;
    // cout << "units at first" << endl;
    // cout << units << endl;
    uvec A = find(units > U);
    // cout << "A" << endl;
    // cout << A << endl;
    units.elem(A) = epsilon*(units.elem(A) - U);
    //units(start_unit) = epsilon*(units(start_unit) - U);
    //A = {start_unit};
    while(!A.is_empty()) {
      avalanche_size += A.n_elem;
      avalanche_duration++;
      // cout << avalanche_size << " " << avalanche_duration << endl;
      if(avalanche_duration > max_duration_factor*units.n_elem) {
        cout << "TOO BIG I QUIT!" << endl;
        throw SupcritException();
      }
      int_func(units,A);
      if(clamp_neg) units.elem(find(units < 0)).zeros(); // reset negative values to zero
      Avec = conv_to<vector<inds_t>>::from(move(A));
      avalanches.insert(avalanches.end(),make_move_iterator(Avec.begin()),make_move_iterator(Avec.end()));
      // cout << "resetted " << endl << units << endl;
      A = find(units > U);
      // cout << "new A " << endl << A << endl;
      if(!A.is_empty()) avalanches.push_back(STEP_SEP);
      units.elem(A) = epsilon*(units.elem(A) - U);
    }
    avs_sizes.push_back(avalanche_size);
    avs_durations.push_back(avalanche_duration);
  }


  template <typename Func>
  void simulate_model(vec &units,unsigned int numAvalanches,double deltaU,
                      const Col<inds_t> &external_weights,bool num_avs,Func int_func) {
    py::gil_scoped_release release;
    avalanches = vector<inds_t>();
    avs_inds = vector<unsigned int>();
    avs_inds.reserve(numAvalanches);
    avalanches.reserve(10*numAvalanches); //reserve least amount of space needed
    avs_sizes = vector<unsigned int>();
    avs_sizes.reserve(numAvalanches);
    avs_durations = vector<unsigned int>();
    avs_durations.reserve(numAvalanches);
    activated_units = vector<inds_t>();
    activated_units.reserve(numAvalanches);
    act_inds = vector<unsigned int>();
    act_inds.reserve(numAvalanches);
    if(use_old_units) {
      units = old_units;
    }
    vector<vec> uh;
    std::random_device rd;  
    std::mt19937 gen(rd());
    std::discrete_distribution<> dis(external_weights.begin(),external_weights.end());
    unsigned int avalanche_counter=0;
    unsigned int r;
    act_inds.push_back(0);
    // num_avs == false means num_steps
    while((num_avs && (avalanche_counter < numAvalanches))  | (!num_avs && activated_units.size() < numAvalanches)) {
      r = dis(gen);
      activated_units.push_back(r);
      units = tau*units;
      units(r) += deltaU*(random ? randu<double>() :1);  
      if(units(r) >= U) {
        act_inds.push_back(activated_units.size());
        avs_inds.push_back(avalanches.size());
        try {
          handle_avalanche(units,int_func,epsilon,U,avalanches,avs_sizes,avs_durations);
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
      if(record_units && avalanche_counter > 100) {
        uh.push_back(units);
      }
    }
    if(record_units) {
      unit_hist = mat(uh.size(),units.n_elem);//units.n_elem);
      for(unsigned int i = 0; i < uh.size();i++) {
        unit_hist.row(i) = uh[i].t();
      }
    }
    avs  = arma::Mat<inds_t>(avalanches.data(),avalanches.size(),1,false,true);
    inds = arma::Mat<unsigned int>(avs_inds.data(),avs_inds.size(),1,false,true);
    old_units = units;
  }


  void ergodic_sample(const vec &starting_units,const mat &torus_coords, const mat &W) {
    py::gil_scoped_release release;
    // setup storage vectors
    auto numAvalanches = starting_units.n_rows;
    avalanches = vector<inds_t>();
    avs_inds = vector<unsigned int>();
    avalanches.reserve(10*numAvalanches); //reserve least amount of space needed
    avs_sizes = vector<unsigned int>();
    avs_sizes.reserve(numAvalanches);
    avs_durations = vector<unsigned int>();
    avs_durations.reserve(numAvalanches);
    activated_units = vector<inds_t>();
    activated_units.reserve(numAvalanches);
    act_inds = vector<unsigned int>();
    act_inds.reserve(numAvalanches);
    vector<vec> uh;
    act_inds.push_back(0);
    auto a_dump = avalanches;
    auto as_dump = avs_sizes;
    auto ad_dump = avs_durations;
    auto deltaU = 1-sum(W,1).max()-1e-10; 
    auto int_func = [&](vec &units,uvec &inds) {
                     vec A(size(units),fill::zeros);
                     A.elem(inds) += 1;
                     units += W*A;};
    // cout << "hi" << endl;                 
    mat id = mat(size(W),fill::eye);
    mat transformation = id - W; 
    vec offset = sum(W,1);
    vec units;
    for(unsigned int i = 0; i < numAvalanches; i++)  {
      // cout << "i " << i << endl;
      units = torus_coords.row(i).t();
      units(starting_units(i)) = 0; 
      // cout << "units " << endl << units << endl;
      vec transformed = (transformation * units) + offset;
      handle_avalanche(transformed,int_func,epsilon,U,a_dump,as_dump,ad_dump);
      // cout << "new units " << endl << units << endl;
      if(record_units) {
        uh.push_back(transformed);
      }
      transformed(starting_units(i)) = 1+deltaU/2;
      handle_avalanche(transformed,int_func,epsilon,U,avalanches,avs_sizes,avs_durations);
      avs_inds.push_back(avalanches.size());
    }
    if(record_units) {
      unit_hist = mat(uh.size(),units.n_elem);
      for(unsigned int i = 0; i < uh.size();i++) {
        unit_hist.row(i) = uh[i].t();
      }
    }
  }


  void ergodic_sample(unsigned int numAvalanches, const mat &W) {
        py::gil_scoped_release release;
    // setup storage vectors
    avalanches = vector<inds_t>();
    avs_inds = vector<unsigned int>();
    avalanches.reserve(10*numAvalanches); //reserve least amount of space needed
    avs_sizes = vector<unsigned int>();
    avs_sizes.reserve(numAvalanches);
    avs_durations = vector<unsigned int>();
    avs_durations.reserve(numAvalanches);
    activated_units = vector<inds_t>();
    activated_units.reserve(numAvalanches);
    act_inds = vector<unsigned int>();
    act_inds.reserve(numAvalanches);
    vector<vec> uh;
    act_inds.push_back(0);
    auto a_dump = avalanches;
    auto as_dump = avs_sizes;
    auto ad_dump = avs_durations;
    auto deltaU = 1-sum(W,1).max()-1e-10; 
    auto int_func = [&](vec &units,uvec &inds) {
                     vec A(size(units),fill::zeros);
                     A.elem(inds) += 1;
                     units += W*A;};
    // cout << "hi" << endl;
    auto external_weights = Col<inds_t>(W.n_rows,fill::ones);
    std::random_device rd;  
    std::mt19937 gen(rd());
    std::discrete_distribution<> dis(external_weights.begin(),external_weights.end());
    int avalanche_counter = 0;
    mat id = mat(size(W),fill::eye);
    mat transformation = id-W;
    vec offset = sum(W,1);
    for(unsigned int i = 0; i < numAvalanches; i++)  {
      // cout << "i " << i << endl;

      auto starting_unit = dis(gen);
      activated_units.push_back(starting_unit);
      colvec units = randu<colvec>(W.n_rows);
      units(starting_unit) = 0;
      units = (transformation * units) + offset;
      handle_avalanche(units,int_func,epsilon,U,a_dump,as_dump,ad_dump);
      units(starting_unit) = 1+deltaU/2; 
      act_inds.push_back(activated_units.size());
      try {
        handle_avalanche(units,int_func,epsilon,U,avalanches,avs_sizes,avs_durations);
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
      avs_inds.push_back(avalanches.size());
    }
  }

/*
    template <typename Func>
  void handle_avalanche_uncorrelated(vec  &units,
                        Func &int_func, double epsilon,int U,
                        vector<inds_t> &avalanches,
                        vector<unsigned int> &avs_sizes,
                        vector<unsigned int> &avs_durations) {
    vector<inds_t> Avec;
    unsigned int avalanche_duration = 0;
    unsigned int avalanche_size = 0;
    // cout << "units at first" << endl;
    // cout << units << endl;
    uvec A = find(units > U);
    // cout << "A" << endl;
    // cout << A << endl;
    units.elem(A) = 0;//randu<uvec>(A.n_elem);//(epsilon*(units.elem(A) - U);
    //units(start_unit) = epsilon*(units(start_unit) - U);
    //A = {start_unit};
    uvec allA = A;
    while(!A.is_empty()) {
      avalanche_size += A.n_elem;
      avalanche_duration++;
      // cout << avalanche_size << " " << avalanche_duration << endl;
      if(avalanche_duration > max_duration_factor*units.n_elem) {
        cout << "TOO BIG I QUIT!" << endl;
        throw SupcritException();
      }
      int_func(units,A);
      units.elem(allA).zeros();
      if(clamp_neg) units.elem(find(units < 0)).zeros(); // reset negative values to zero
      Avec = conv_to<vector<inds_t>>::from(move(A));
      avalanches.insert(avalanches.end(),make_move_iterator(Avec.begin()),make_move_iterator(Avec.end()));
      // cout << "resetted " << endl << units << endl;
      A = find(units > U);
      allA = join_cols(allA,A);
      // cout << "new A " << endl << A << endl;
      if(!A.is_empty()) avalanches.push_back(STEP_SEP);
      units.elem(A) = 0;//epsilon*(units.elem(A) - U);
    }
    units.elem(allA) = deltaU*randu<uvec>(allA.n_elem);
    avs_sizes.push_back(avalanche_size);
    avs_durations.push_back(avalanche_duration);
  }


  template <typename Func>
  void simulate_uncorrelated(vec &units,unsigned int numAvalanches,double deltaU,
                      const Col<inds_t> &external_weights,bool num_avs,Func int_func) {
    py::gil_scoped_release release;
    avalanches = vector<inds_t>();
    avs_inds = vector<unsigned int>();
    avs_inds.reserve(numAvalanches);
    avalanches.reserve(10*numAvalanches); //reserve least amount of space needed
    avs_sizes = vector<unsigned int>();
    avs_sizes.reserve(numAvalanches);
    avs_durations = vector<unsigned int>();
    avs_durations.reserve(numAvalanches);
    activated_units = vector<inds_t>();
    activated_units.reserve(numAvalanches);
    act_inds = vector<unsigned int>();
    act_inds.reserve(numAvalanches);
    if(use_old_units) {
      units = old_units;
    }
    vector<vec> uh;
    std::random_device rd;  
    std::mt19937 gen(rd());
    std::discrete_distribution<> dis(external_weights.begin(),external_weights.end());
    unsigned int avalanche_counter=0;
    unsigned int r;
    act_inds.push_back(0);
    // num_avs == false means num_steps
    while((num_avs && (avalanche_counter < numAvalanches))  | (!num_avs && activated_units.size() < numAvalanches)) {
      r = dis(gen);
      activated_units.push_back(r);
      units = tau*units;
      units(r) += deltaU*(random ? randu<double>() :1);  
      if(units(r) >= U) {
        act_inds.push_back(activated_units.size());
        avs_inds.push_back(avalanches.size());
        try {
          handle_avalanche_uncorrelated(units,int_func,epsilon,U,avalanches,avs_sizes,avs_durations);
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
      if(record_units && avalanche_counter > 100) {
        uh.push_back(units);
      }
    }
    if(record_units) {
      unit_hist = mat(uh.size(),units.n_elem);//units.n_elem);
      for(unsigned int i = 0; i < uh.size();i++) {
        unit_hist.row(i) = uh[i].t();
      }
    }
    avs  = arma::Mat<inds_t>(avalanches.data(),avalanches.size(),1,false,true);
    inds = arma::Mat<unsigned int>(avs_inds.data(),avs_inds.size(),1,false,true);
    old_units = units;
    }*/


  void sample_uncorrelated(unsigned int numAvalanches, const mat &W) {
    py::gil_scoped_release release;
    // setup storage vectors
    avalanches = vector<inds_t>();
    avs_inds = vector<unsigned int>();
    avalanches.reserve(10*numAvalanches); //reserve least amount of space needed
    avs_sizes = vector<unsigned int>();
    avs_sizes.reserve(numAvalanches);
    avs_durations = vector<unsigned int>();
    avs_durations.reserve(numAvalanches);
    activated_units = vector<inds_t>();
    activated_units.reserve(numAvalanches);
    act_inds = vector<unsigned int>();
    act_inds.reserve(numAvalanches);
    vector<vec> uh;
    act_inds.push_back(0);
    auto a_dump = avalanches;
    auto as_dump = avs_sizes;
    auto ad_dump = avs_durations;
    auto deltaU = 1-sum(W,1).max()-1e-10; 
    auto int_func = [&](vec &units,uvec &inds) {
                     vec A(size(units),fill::zeros);
                     A.elem(inds) += 1;
                     units += W*A;};
    // cout << "hi" << endl;
    auto external_weights = Col<inds_t>(W.n_rows,fill::ones);
    std::random_device rd;  
    std::mt19937 gen(rd());
    std::discrete_distribution<> dis(external_weights.begin(),external_weights.end());
    int avalanche_counter = 0; 
    for(unsigned int i = 0; i < numAvalanches; i++)  {
      // cout << "i " << i << endl;

      auto starting_unit = dis(gen);
      activated_units.push_back(starting_unit);
      colvec units = randu<colvec>(W.n_rows);
      units(starting_unit) = 1-1e-10+deltaU;
      act_inds.push_back(activated_units.size());
      try {
        handle_avalanche(units,int_func,epsilon,U,avalanches,avs_sizes,avs_durations);
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
      avs_inds.push_back(avalanches.size());
    }
  }


  void sample_uncorrelated(unsigned int numAvalanches, const mat &W,double deltaU) {
    py::gil_scoped_release release;
    // setup storage vectors
    avalanches = vector<inds_t>();
    avs_inds = vector<unsigned int>();
    avalanches.reserve(10*numAvalanches); //reserve least amount of space needed
    avs_sizes = vector<unsigned int>();
    avs_sizes.reserve(numAvalanches);
    avs_durations = vector<unsigned int>();
    avs_durations.reserve(numAvalanches);
    activated_units = vector<inds_t>();
    activated_units.reserve(numAvalanches);
    act_inds = vector<unsigned int>();
    act_inds.reserve(numAvalanches);
    vector<vec> uh;
    act_inds.push_back(0);
    auto a_dump = avalanches;
    auto as_dump = avs_sizes;
    auto ad_dump = avs_durations;
    //auto deltaU = 1-sum(W,1).max()-1e-10; 
    auto int_func = [&](vec &units,uvec &inds) {
                     vec A(size(units),fill::zeros);
                     A.elem(inds) += 1;
                     units += W*A;};
    // cout << "hi" << endl;
    auto external_weights = Col<inds_t>(W.n_rows,fill::ones);
    std::random_device rd;  
    std::mt19937 gen(rd());
    std::discrete_distribution<> dis(external_weights.begin(),external_weights.end());
    int avalanche_counter = 0; 
    for(unsigned int i = 0; i < numAvalanches; i++)  {
      // cout << "i " << i << endl;

      auto starting_unit = dis(gen);
      activated_units.push_back(starting_unit);
      colvec units = randu<colvec>(W.n_rows);
      units(starting_unit) = 1-1e-10+deltaU;
      act_inds.push_back(activated_units.size());
      try {
        handle_avalanche(units,int_func,epsilon,U,avalanches,avs_sizes,avs_durations);
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
      avs_inds.push_back(avalanches.size());
    }
  }  

  void sample_uncorrelated_grid(unsigned int numAvalanches, double weight,const mat &conns) {
    py::gil_scoped_release release;
    // setup storage vectors
    avalanches = vector<inds_t>();
    avs_inds = vector<unsigned int>();
    avalanches.reserve(10*numAvalanches); //reserve least amount of space needed
    avs_sizes = vector<unsigned int>();
    avs_sizes.reserve(numAvalanches);
    avs_durations = vector<unsigned int>();
    avs_durations.reserve(numAvalanches);
    activated_units = vector<inds_t>();
    activated_units.reserve(numAvalanches);
    act_inds = vector<unsigned int>();
    act_inds.reserve(numAvalanches);
    vector<vec> uh;
    act_inds.push_back(0);
    auto a_dump = avalanches;
    auto as_dump = avs_sizes;
    auto ad_dump = avs_durations;
    auto deltaU = 1-weight*conns.n_cols-1e-10; 
    auto int_func = [&](vec &units,uvec &inds) {
                     for (auto i : inds) {
                       //cout << conv_to<uvec>::from(conns.row(i)) << endl;
                       units.elem(conv_to<uvec>::from(conns.row(i)))+=weight;
                     }};
    // cout << "hi" << endl;
    auto external_weights = Col<inds_t>(conns.n_rows,fill::ones);
    std::random_device rd;  
    std::mt19937 gen(rd());
    std::discrete_distribution<> dis(external_weights.begin(),external_weights.end());
    int avalanche_counter = 0; 
    for(unsigned int i = 0; i < numAvalanches; i++)  {
      // cout << "i " << i << endl;

      auto starting_unit = dis(gen);
      activated_units.push_back(starting_unit);
      colvec units = randu<colvec>(conns.n_rows);
      units(starting_unit) = 1-1e-10+deltaU;
      act_inds.push_back(activated_units.size());
      try {
        handle_avalanche(units,int_func,epsilon,U,avalanches,avs_sizes,avs_durations);
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
      avs_inds.push_back(avalanches.size());
    }
  }  

  void simulate_BP(int num_avs,const mat &W) {
    auto external_weights = Col<inds_t>(W.n_rows,fill::ones);
    std::random_device rd;  
    std::mt19937 gen(rd());
    std::discrete_distribution<> dis(external_weights.begin(),external_weights.end());
    avs_inds = vector<unsigned int>();
    avalanches = vector<inds_t>();
    avs_inds.reserve(num_avs);
    avalanches.reserve(10*num_avs);
    avs_inds.push_back(0);
    cout << "hi" << endl;
    for(int idx=1; idx<=num_avs;idx++) {
      Col<inds_t> states(W.n_rows,fill::zeros); //0-> inactive 1 active 2 refractory
      auto r1 = dis(gen);
      states(r1) = 1;
      uvec currently_active = find(states==1);
      avalanches.push_back(r1);
      int d = 0;
      int size = 1; 
      while (!currently_active.is_empty() &&
             (d++ < max_duration_factor*W.n_rows) ) {
        vec A(W.n_rows,fill::zeros);
        A.elem(currently_active) +=1;
        //cout << "before probs "<< endl;
        vec probs = W*A;
        //cout << "probs " << probs;
        for(int i=0; i<W.n_rows;i++) {
          if(probs(i)>0) {
            if (states(i)==1) {
              states(i) = 0;
            } else if (randu()<probs(i)) {
              states(i) = 1;
              avalanches.push_back(i);
              size += 1;
             }
            else {
              states(i) = 0;
            } 
          }
        }
        currently_active = find(states==1);
        if(!currently_active.is_empty()) avalanches.push_back(STEP_SEP);
      }
      avs_inds.push_back(avalanches.size());
    }
    avs  = arma::Mat<inds_t>(avalanches.data(),avalanches.size(),1,false,true);
    inds = arma::Mat<unsigned int>(avs_inds.data(),avs_inds.size(),1,false,true);

  }



  void simulate_KC(int num_avs,const mat &W) {
    auto external_weights = Col<inds_t>(W.n_rows,fill::ones);
    std::random_device rd;  
    std::mt19937 gen(rd());
    //std::discrete_distribution<> dis(external_weights.begin(),external_weights.end());
    std::uniform_int_distribution<> dis(0,W.n_rows-1);
    avs_inds = vector<unsigned int>();
    avalanches = vector<inds_t>();
    avs_inds.reserve(num_avs);
    avalanches.reserve(10*num_avs);
    avs_inds.push_back(0);
    act_inds = vector<unsigned int>();
    act_inds.reserve(num_avs);
    cout << "hi" << endl;
    Col<inds_t> states(W.n_rows,fill::zeros); //0-> inactive 1 active 2 refractory
    for(int idx=1; idx<=num_avs;idx++) {
      states.fill(0);
      auto r1 = dis(gen);
      states(r1) = 1;
      activated_units.push_back(r1);
      uvec currently_active = find(states==1);
      avalanches.push_back(r1);
      act_inds.push_back(activated_units.size());
       while (!currently_active.is_empty()) {
        vec A(W.n_rows,fill::zeros);
        A.elem(currently_active) +=1;
        //cout << "before probs "<< endl;
        vec probs = W*A;
        //cout << "probs " << probs;
        for(int i=0; i<W.n_rows;i++) {
          if(states(i) == 1) {
            states(i) = 2;
          } else if ((states(i) == 0) && (probs(i)>0)) {
            if (randu()<probs(i)) {
            states(i) =  1 ;
            avalanches.push_back(i);
            }
          }
        }
        currently_active = find(states==1);

        if(!currently_active.is_empty()) avalanches.push_back(STEP_SEP);
      }
      avs_inds.push_back(avalanches.size());
    }
    avs  = arma::Mat<inds_t>(avalanches.data(),avalanches.size(),1,false,true);
    inds = arma::Mat<unsigned int>(avs_inds.data(),avs_inds.size(),1,false,true);

  }

  void simulate_KC_grid(int num_avs,double weight,const mat &conns) {
    auto external_weights = Col<inds_t>(conns.n_rows,fill::ones);
    std::random_device rd;  
    std::mt19937 gen(rd());
    std::discrete_distribution<> dis(external_weights.begin(),external_weights.end());
    avs_inds = vector<unsigned int>();
    avalanches = vector<inds_t>();
    avs_inds.reserve(num_avs);
    avalanches.reserve(10*num_avs);
    avs_inds.push_back(0);
    cout << "hi" << endl;
    for(int idx=1; idx<=num_avs;idx++) {
      Col<inds_t> states(conns.n_rows,fill::zeros); //0-> inactive 1 active 2 refractory
      auto r1 = dis(gen);
      states(r1) = 1;
      uvec currently_active = find(states==1);
      avalanches.push_back(r1);
       while (!currently_active.is_empty()) {
         vec probs(conns.n_rows,fill::zeros); 
         for (auto i : currently_active) {
           //cout << conv_to<uvec>::from(conns.row(i)) << endl;
           probs.elem(conv_to<uvec>::from(conns.row(i)))+=weight;
         }
         //vec A(W.n_rows,fill::zeros);
         //A.elem(currently_active) +=1;
        //cout << "before probs "<< endl;
        //vec probs = W*A;
        //cout << "probs " << probs;
        for(int i=0; i<conns.n_rows;i++) {
          if(states(i) == 1) {
            states(i) = 2;
          } else if ((states(i) == 0) && (probs(i)>0)) {
            if (randu()<probs(i)) {
            states(i) =  1 ;
            avalanches.push_back(i);
            }
          }
        }
        currently_active = find(states==1);
        if(!currently_active.is_empty()) avalanches.push_back(STEP_SEP);
      }
      avs_inds.push_back(avalanches.size());
    }
    avs  = arma::Mat<inds_t>(avalanches.data(),avalanches.size(),1,false,true);
    inds = arma::Mat<unsigned int>(avs_inds.data(),avs_inds.size(),1,false,true);

  }  




  void simulate_KC_grid(int num_avs,double weight,const vector<uvec> &conns,int starting_unit) {
    auto external_weights = Col<inds_t>(conns.size(),fill::ones);
    std::random_device rd;  
    std::mt19937 gen(rd());
    std::discrete_distribution<> dis(external_weights.begin(),external_weights.end());
    avs_inds = vector<unsigned int>();
    avalanches = vector<inds_t>();
    avs_inds.reserve(num_avs);
    avalanches.reserve(10*num_avs);
    avs_inds.push_back(0);
    cout << "hi" << endl;
    for(int idx=1; idx<=num_avs;idx++) {
      Col<inds_t> states(conns.size(),fill::zeros); //0-> inactive 1 active 2 refractory
      auto r1 = starting_unit;//dis(gen);
      states(r1) = 1;
      uvec currently_active = find(states==1);
      avalanches.push_back(r1);
       while (!currently_active.is_empty()) {
         vec probs(conns.size(),fill::zeros); 
         for (auto i : currently_active) {
           //cout << conv_to<uvec>::from(conns.row(i)) << endl;
           probs.elem(conv_to<uvec>::from(conns[i]))+=weight;
         }
         //vec A(W.n_rows,fill::zeros);
         //A.elem(currently_active) +=1;
        //cout << "before probs "<< endl;
        //vec probs = W*A;
        //cout << "probs " << probs;
       for(int i=0; i<conns.size();i++) {
          if(states(i) == 1) {
            states(i) = 2;
          } else if ((states(i) == 0) && (probs(i)>0)) {
            if (randu()<probs(i)) {
            states(i) =  1 ;
            avalanches.push_back(i);
            }
          }
        }
        currently_active = find(states==1);
        if(!currently_active.is_empty()) avalanches.push_back(STEP_SEP);
      }
      avs_inds.push_back(avalanches.size());
    }
    avs  = arma::Mat<inds_t>(avalanches.data(),avalanches.size(),1,false,true);
    inds = arma::Mat<unsigned int>(avs_inds.data(),avs_inds.size(),1,false,true);

  }  

  void simulate_model_mat(vec &units, int numAvalanches,const mat &W,double deltaU,Col<inds_t> &external_weights) {
    simulate_model(units,numAvalanches,deltaU,external_weights,true,
                   [&](vec &units,uvec &inds) {
                     vec A(size(units),fill::zeros);
                     A.elem(inds) += 1;
                     units += W*A;});
  }


  void simulate_model_mat_duration(vec &units, int num_steps,const mat &W,double deltaU,Col<inds_t> &external_weights) {
    simulate_model(units,num_steps,deltaU, external_weights,false,
                   [&](vec &units,uvec &inds) {
                     vec A(size(units),fill::zeros);
                     A.elem(inds) += 1;
                     units += W*A;});
  }


  void simulate_model_mat(vec &units, int numAvalanches,const mat &W,double deltaU) {
    auto external_weights = Col<inds_t>(units.n_elem,fill::ones);
    simulate_model_mat(units,numAvalanches,W,deltaU, external_weights);
  }

  void simulate_model_const(vec &units, int numAvalanches,double w,double deltaU) {
    auto external_weights = Col<inds_t>(units.n_elem,fill::ones);
    simulate_model(units,numAvalanches,deltaU, external_weights,true,
                   [=](vec &units,uvec &inds) { units += w*inds.n_elem;});
  }


  void simulate_model_const_duration(vec &units, int num_steps,double w,double deltaU) {
    auto external_weights = Col<inds_t>(units.n_elem,fill::ones);
    simulate_model(units,num_steps,deltaU, external_weights,false,
                   [=](vec &units,uvec &inds) { units += w*inds.n_elem;});
  }




  void simulate_model_conv(vec &units, int numAvalanches,const vec &W,double deltaU) {
    const cx_vec Wfft = fft(W);
    auto external_weights = Col<inds_t>(units.n_elem,fill::ones);
    simulate_model(units,numAvalanches,deltaU,external_weights,true,
                   [&](vec &units,uvec &inds) {
                     vec A(size(units),fill::zeros);
                     A.elem(inds) += 1;
                     units += real(ifft(Wfft%fft(A))); });
  }


  tuple<vector<int>,vector<int>>
  get_avs_size_and_duration(Col<inds_t> &avs,vector<unsigned int> &avs_inds) {
    py::gil_scoped_release release;
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


  void simulate_model_grid(vec &units, int numAvalanches,double weight,const mat &conns,double deltaU) {
    auto external_weights = Col<inds_t>(units.n_elem,fill::ones);
    simulate_model(units,numAvalanches,deltaU,external_weights,true,
                   [&](vec &units,uvec &inds) {
                     for (auto i : inds) {
                       //cout << conv_to<uvec>::from(conns.row(i)) << endl;
                       units.elem(conv_to<uvec>::from(conns.row(i)))+=weight;
                     }});
  }  

  tuple<vector<int>,vector<int>> get_avs_size_and_duration() {
    return get_avs_size_and_duration(avs,avs_inds);
  }

  vector<vector<vector<inds_t>>>
  get_spiking_patterns(Col<inds_t> & avs, vector<unsigned int> &avs_inds) {
    py::gil_scoped_release release;
    vector<vector<vector<inds_t>>> spiking_patterns;
    spiking_patterns.reserve(avs_inds.size());
    for(unsigned int i=0; i < avs_inds.size()-1;i++) {
      vector<vector<inds_t>> sp;
      vector<inds_t> cur_step;
      int ub = avs_inds.size();
      if(i < avs_inds.size()-1) {
        ub = avs_inds[i+1];
       }
      for(unsigned int j=avs_inds[i];j < ub;j++) {
        if(avs[j] != STEP_SEP) {
          cur_step.push_back(avs[j]);
        } else {
          if(!cur_step.empty()) sp.push_back(move(cur_step));
          cur_step.clear();
        }
      }
      if(!cur_step.empty()) sp.push_back(move(cur_step));
      cur_step.clear();
      spiking_patterns.push_back(move(sp));
      sp.clear();
    }
    return spiking_patterns;
  }

  vector<vector<vector<inds_t>>> get_spiking_patterns() {
    return get_spiking_patterns(avs,avs_inds);
  }

  tuple<Col<inds_t>,vector<unsigned int>>
  subnetwork_avalanches(Col<inds_t> &avs,vector<unsigned int> &avs_inds,
                        set<inds_t> &subnet_inds) {
    py::gil_scoped_release release;
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
    }
    if(subavs_inds.back() >= subavs.size()) {
      subavs_inds.pop_back(); //remove trailing avs of size 0
    }
    return make_tuple(conv_to<Col<inds_t>>::from(subavs),subavs_inds);
  }

  tuple<Col<inds_t>,vector<unsigned int>>
  subnetwork_avalanches(set<inds_t> &subnet_inds) {
    return subnetwork_avalanches(avs,avs_inds,subnet_inds);
  }

};

void test(mat & test) {
  std::cout << "Hi there from c++" << std::endl;
}

using EHE = EHE__module_suffix__;

PYBIND11_PLUGIN(ehe_detailed__module_suffix__) {

  py::module m("ehe", "Simulate ehe model");
  py::class_<EHE>(m, "EHE")
    .def(py::init<>())
    .def("ergodic_sample",(void (EHE::*)(const vec&,const mat&,const mat&)) &EHE::ergodic_sample)
    .def("ergodic_sample",(void (EHE::*)(unsigned int, const mat&)) &EHE::ergodic_sample)
    .def("simulate_model_mat",(void (EHE::*)(vec&,int,const mat&,double)) &EHE::simulate_model_mat)
    .def("simulate_model_mat",(void (EHE::*)(vec&,int,const mat&,double,Col<inds_t>&)) &EHE::simulate_model_mat)
    .def("simulate_model_conv",(void (EHE::*)(vec&,int,const vec&,double)) & EHE::simulate_model_conv)
    .def("simulate_model_grid",(void (EHE::*)(vec&,int,double,const mat&,double)) &EHE::simulate_model_grid)
    .def("simulate_KC",(void (EHE::*)(int,const mat&)) &EHE::simulate_KC)
    .def("simulate_BP",(void (EHE::*)(int,const mat&)) &EHE::simulate_BP)
    .def("simulate_KC_grid",(void (EHE::*)(int,double,const mat&)) &EHE::simulate_KC_grid)
    .def("simulate_KC_grid",(void (EHE::*)(int,double,const vector<uvec>&,int)) &EHE::simulate_KC_grid)
    .def("sample_uncorrelated_grid",(void (EHE::*)(int,double,const mat&)) &EHE::sample_uncorrelated_grid)
    .def("sample_uncorrelated",(void (EHE::*)(unsigned int,const mat&)) &EHE::sample_uncorrelated)
    .def("sample_uncorrelated",(void (EHE::*)(unsigned int,const mat&,double)) &EHE::sample_uncorrelated) 
    .def("simulate_model_mat_duration",(void (EHE::*)(vec&,int,const mat&,double,Col<inds_t>&)) &EHE::simulate_model_mat_duration)
    .def("simulate_model_const",&EHE::simulate_model_const)
    .def("simulate_model_const_duration",&EHE::simulate_model_const_duration)    
    .def("get_avs_size_and_duration",(tuple<vector<int>,vector<int>> (EHE::*)(void)) &EHE::get_avs_size_and_duration)
    .def("get_avs_size_and_duration",
         (tuple<vector<int>,vector<int>> (EHE::*)(Col<inds_t>&,vector<unsigned int>&)) &EHE::get_avs_size_and_duration)
    .def("subnetwork_avalanches",
         (tuple<Col<inds_t>,vector<unsigned int>> (EHE::*)(Col<inds_t>&,vector<unsigned int>&,set<inds_t>&))
         &EHE::subnetwork_avalanches)
    .def("subnetwork_avalanches",
         (tuple<Col<inds_t>,vector<unsigned int>> (EHE::*)(set<inds_t>&)) &EHE::subnetwork_avalanches)
    .def("get_spiking_patterns", (vector<vector<vector<inds_t>>> (EHE::*)(void)) &EHE::get_spiking_patterns)
    .def("get_spiking_patterns", (vector<vector<vector<inds_t>>> (EHE::*)(Col<inds_t>&,vector<unsigned int>&))
         &EHE::get_spiking_patterns)
    .def_readwrite("avalanches",&EHE::avalanches)
    .def_readwrite("avs",&EHE::avs)
    .def_readwrite("unit_hist",&EHE::unit_hist)
    .def_readwrite("clamp_neg",&EHE::clamp_neg)
    .def_readwrite("record_units",&EHE::record_units)
    .def_readwrite("activated_units",&EHE::activated_units)
    .def_readwrite("epsilon",&EHE::epsilon)
    .def_readwrite("tau",&EHE::tau)
    .def_readwrite("random",&EHE::random)
    .def_readwrite("act_inds",&EHE::act_inds)
    .def_readwrite("avs_inds",&EHE::avs_inds)
    .def_readwrite("avs_durations",&EHE::avs_durations)
    .def_readwrite("max_duration_factor",&EHE::max_duration_factor)
    .def_readwrite("use_old_units",&EHE::use_old_units)
    .def_readwrite("old_units",&EHE::old_units)
    .def_readwrite("avs_sizes",&EHE::avs_sizes);

  m.def("test",&test,"test");

  return m.ptr();

}
