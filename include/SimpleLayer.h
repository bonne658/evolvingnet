#ifndef EVOLV_SIMPLE_LAYER_H
#define EVOLV_SIMPLE_LAYER_H
#include <iostream>
#include <cmath>
#include <ctime>
#include "Layer.h"

class SimpleLayer:public Layer {
public:
  SimpleLayer(int ic, int oc, float lr,float k);
  virtual void Forward(vector<float> input);
  virtual void BackWard(vector<float> delta);
  float Sigmoid(float z) {
    return 1/(1+ exp(-z));
  }
};

SimpleLayer::SimpleLayer(int ic, int oc, float lr, float k) {
  ic_ = ic;
  oc_ = oc;
  lr_ = lr;
  bias_.resize(oc, 0.6);
  bias_[0] = k;
  srand((unsigned)time(NULL));
	for (size_t i = 0; i < oc; ++i) {
    bias_[i] = rand() % 100 / 100.0;
    if (rand() % 2) bias_[i] = -bias_[i];
  }
  // w11,w12...w1n,w21,w22...w2n...
  weight_.resize(ic*oc);
  srand((unsigned)time(NULL));
	for (size_t i = 0; i < ic*oc; ++i) {
    weight_[i] = rand() % 100 / 100.0;
    if (rand() % 2) weight_[i] = -weight_[i];
  }
  // weight_[0] = 0.15;
  // weight_[1] = 0.2;
  // weight_[2] = 0.25;
  // weight_[3] = 0.3;
  // weight_[4] = 0.4;
  // weight_[5] = 0.45;
  // weight_[6] = 0.5;
  // weight_[7] = 0.55;
  input_.resize(ic);
  output_.resize(oc);
  delta_.resize(ic_, 0);
  // for(int i = 0; i< ic*oc; ++i) {
  //   weight_[i] = 0.12 + 0.07 * i;
  // }
}

void SimpleLayer::Forward(vector<float> input) {
  // std::cout << "*******Forward*******\n";
  for(int i = 0; i < oc_; ++i) {
    output_[i] = 0;
  }
  for(int i = 0; i < input.size(); ++i) {
    input_[i] = input[i];
    // std::cout << input_[i] << endl;
  }
  for(int i = 0; i < oc_; ++i) {
    for(int j = 0; j < ic_; ++j) {
      output_[i] += input[j] * weight_[i*ic_+j];
      // std::cout << input[j] << " * " << weight_[i*ic_+j] << endl;
      // std::cout << "sum = " << output_[i] << endl;
    }
    output_[i] += bias_[i];
    // output_[i] += 0.33;
    output_[i] = Sigmoid(output_[i]);
    // std::cout << "add bias: " << output_[i] << endl;
    // std::cout << "after sigmoid: " << output_[i] << endl;
  }
  // std::cout << "after : " << endl;
}

void SimpleLayer::BackWard(vector<float> delta) {
  for(int j = 0; j < ic_; ++j) {
    delta_[j] = 0;
  }
  for(int j = 0; j < ic_; ++j) {
    for(int i = 0; i < oc_; ++i) {
      delta_[j] += weight_[i*ic_+j]*delta[i];
      // std::cout << delta[j] << endl;
    }
  }
  // std::cout << "*******Backward*******\n";
  for(int i = 0; i < oc_; ++i) {
    for(int j = 0; j < ic_; ++j) {
      float tmp = input_[j]*delta[i];
      weight_[i*ic_+j] -= lr_ * tmp;
      // std::cout << weight_[i*ic_+j] << endl;
    }
    bias_[i] -= lr_ * delta[i];
    // std::cout << bias_[i] << endl;
  }
}
#endif

