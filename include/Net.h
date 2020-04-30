#ifndef EVOLV_NET_H
#define EVOLV_NET_H
#include <vector>
#include <iostream>
#include <ctime>
#include "Layer.h"
#include "SimpleLayer.h"

using namespace std;

class Net {
public:
  int layer_num_;
  int batch_num_;
  float base_learning_rate_;
  float momentum_;
  vector<float> gt_;
  vector<float> input_;
  vector<float> output_;
  vector<float> delta_;
  vector<Layer*> layers_;
  vector<float> data_;
  
  Net();
  void Forward();
  void Backward();
  void Train();
  void Test(float n, float m);
};

Net::Net() {
  input_.resize(2);
  input_.resize(2);
  gt_.resize(2);
  gt_.resize(2);
  data_.resize(1000);
  for(int i = 0; i < 1000; ++i) {
    data_[i] = i/1000.0;
  }
  layer_num_ = 2;
  base_learning_rate_ = 0.5;
  // SimpleLayer tmp1(2, 2, 0.35);
  layers_.push_back(new SimpleLayer(2, 2, base_learning_rate_,1));
  // SimpleLayer tmp2(2, 2, 0.6);
  layers_.push_back(new SimpleLayer(2, 2, base_learning_rate_,2));
}

void Net::Forward() {
  std::cout << "*******Forward*******\n";
  layers_[0]->Forward(input_);
  int i;
  for(i = 1; i < layer_num_; ++i) {
    layers_[i]->Forward(layers_[i-1]->output_);
  }
  output_.resize(layers_[i-1]->output_.size());
  for(int j = 0; j < output_.size(); ++j) {
    output_[j] = layers_[i-1]->output_[j];
    std::cout << output_[j] << endl;
  }
}

void Net::Backward() {
  delta_.resize(2);
  for(int i = 0; i < delta_.size(); ++i) {
    delta_[i] = (output_[i] - gt_[i])*output_[i]*(1-output_[i]);
  }
  layers_[layer_num_-1]->BackWard(delta_);
  int j;
  for(j = layer_num_- 2; j >= 0; --j) {
    layers_[j]->BackWard(layers_[j+1]->delta_);
  }
}

void Net::Train() {
  srand((unsigned)time(NULL));
  while(1) {
    int ran_num = rand() % 1000;
    gt_[0] = data_[ran_num];
    input_[0] = data_[ran_num];
    ran_num = rand() % 1000;
    gt_[1] = data_[ran_num];
    input_[1] = data_[ran_num];
    Forward();
    float loss = 0;
    for(int i = 0; i < 2; ++i) {
      loss += (gt_[i]-output_[i])*(gt_[i]-output_[i]);
    }
    // cout << loss << endl;
    if(loss < 0.0001) {
      break;
    }
    Backward();
  }
}

void Net::Test(float n, float m) {
  input_[0] = n;
  input_[1] = m;
  Forward();
  for(int j = 0; j < output_.size(); ++j) {
    cout << output_[j] << endl;
  }
}
#endif
