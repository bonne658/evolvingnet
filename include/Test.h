#ifndef EVOLV_TEST_H
#define EVOLV_TEST_H
#include <vector>
#include <iostream>
#include "SimpleLayer.h"

using namespace std;

class Test {
public:
  int layer_num_;
  int input_num_;
  int output_num_;
  int hidden_num_;
  float base_learning_rate_;
  vector<vector<float> > gt_;
  vector<vector<float> > input_;
  vector<float> output_;
  vector<Layer*> layers_;
  
  Test();
  void LoadData();
  void Forward(vector<float> in);
  void Backward();
  void Train();
};

Test::Test() {
  output_num_ = 1;
  hidden_num_ = 3;
  input_num_ = 2;
  input_.resize(1);
  gt_.resize(1);
  layer_num_ = 2;
  base_learning_rate_ = 0.5;
  // SimpleLayer tmp1(2, 2, 0.35);
  layers_.push_back(new SimpleLayer(input_num_, hidden_num_, base_learning_rate_, 0.35));
  // SimpleLayer tmp2(2, 2, 0.6);
  layers_.push_back(new SimpleLayer(hidden_num_, output_num_, base_learning_rate_, 0.6));
}

void Test::LoadData() {
  input_[0].resize(2);
  input_[0][0] = 0.05;
  input_[0][1] = 0.1;
  gt_[0].resize(2);
  gt_[0][0] = 0.01;
  gt_[0][1] = 0.99;
}

void Test::Forward(vector<float> in) {
  std::cout << "*******Forward*******\n";
  layers_[0]->Forward(in);
  int i;
  std::cout << "*******haha*******\n";
  cout << layers_[0]->output_.size() << endl;
  for(i = 1; i < layer_num_; ++i) {
    layers_[i]->Forward(layers_[i-1]->output_);
  }
  output_.resize(layers_[i-1]->output_.size());
  for(int j = 0; j < output_.size(); ++j) {
    output_[j] = layers_[i-1]->output_[j];
    std::cout << output_[j] << endl;
  }
}

void Test::Backward() {
  vector<float> delta(output_num_);
  for(int i = 0; i < output_num_; ++i) {
    delta[i] = (output_[i] - gt_[0][i])*output_[i]*(1-output_[i]);
  }
  layers_[layer_num_-1]->BackWard(delta);
  int j;
  for(j = layer_num_- 2; j >= 0; --j) {
    layers_[j]->BackWard(layers_[j+1]->delta_);
  }
}

void Test::Train() {
  LoadData();
  Forward(input_[0]);
  Backward();
}
#endif
