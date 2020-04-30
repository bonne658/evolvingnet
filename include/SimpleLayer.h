#ifndef EVOLV_SIMPLE_LAYER_H
#define EVOLV_SIMPLE_LAYER_H
#include <iostream>
#include <cmath>
#include "Layer.h"

class SimpleLayer:public Layer {
public:
  SimpleLayer(int ic, int oc, float bias, int flag);
  virtual void Forward(vector<float> input);
  virtual void BackWard(vector<float> delta);
  float Sigmoid(float z) {
    return 1/(1+ exp(-z));
  }
};

SimpleLayer::SimpleLayer(int ic, int oc, float lr, int flag) {
  ic_ = ic;
  oc_ = oc;
  lr_ = lr;
  if(flag == 1) bias_.resize(oc, 0.35);
  else bias_.resize(oc, 0.6);
  // w11,w12...w1n,w21,w22...w2n...
  weight_.resize(ic*oc);
  input_.resize(ic);
  output_.resize(oc);
  // for(int i = 0; i< ic*oc; ++i) {
  //   weight_[i] = 0.25 + 0.15 * i;
  // }
  if(flag == 1) {
    weight_[0] = 0.5;
    weight_[1] = 0.51;
    weight_[2] = 0.51;
    weight_[3] = 0.5;
  } else {
    weight_[0] = 0.5;
    weight_[1] = 0.51;
    weight_[2] = 0.51;
    weight_[3] = 0.5;
  }
}

void SimpleLayer::Forward(vector<float> input) {
  // std::cout << "*******Forward*******\n";
  for(int i = 0; i < input.size(); ++i) {
    input_[i] = input[i];
    std::cout << input_[i] << endl;
  }
  for(int i = 0; i < oc_; ++i) {
    for(int j = 0; j < ic_; ++j) {
      output_[i] += input[j] * weight_[i*ic_+j];
      // std::cout << input[j] << " * " << weight_[i*ic_+j] << endl;
      // std::cout << "sum = " << output_[i] << endl;
    }
    output_[i] += bias_[i];
    output_[i] = Sigmoid(output_[i]);
    // std::cout << "add bias: " << output_[i] << endl;
    // std::cout << "after sigmoid: " << output_[i] << endl;
  }
}

void SimpleLayer::BackWard(vector<float> delta) {
  delta_.resize(ic_, 0);
  for(int i = 0; i < ic_; ++i) {
    for(int j = 0; j < oc_; ++j) {
      delta_[i] += weight_[j+i*ic_]*delta[j];
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
  }
}
#endif

