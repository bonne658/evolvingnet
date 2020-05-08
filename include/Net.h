#ifndef EVOLV_NET_H
#define EVOLV_NET_H
#include <vector>
#include <iostream>
#include <fstream>
#include <ctime>
#include "Layer.h"
#include "SimpleLayer.h"

using namespace std;

class Net {
public:
  int layer_num_;
  int batch_num_;
  int input_num_;
  int output_num_;
  int hidden_num_;
  int ran_num_;
  float base_learning_rate_;
  float momentum_;
  vector<vector<float> > gt_;
  vector<vector<float> > input_;
  vector<float> output_;
  vector<float> delta_;
  vector<Layer*> layers_;
  vector<float> data_;
  
  Net();
  void LoadData();
  void Forward(vector<float> in);
  void Backward();
  void Train();
  void Test();
};

Net::Net() {
  output_num_ = 10;
  hidden_num_ = 64;
  input_num_ = 784;
  input_.resize(60000);
  gt_.resize(60000);
  data_.resize(1000);
  for(int i = 0; i < 1000; ++i) {
    data_[i] = i/1000.0;
  }
  layer_num_ = 2;
  base_learning_rate_ = 0.5;
  // SimpleLayer tmp1(2, 2, 0.35);
  layers_.push_back(new SimpleLayer(input_num_, hidden_num_, base_learning_rate_, 0.03));
  // SimpleLayer tmp2(2, 2, 0.6);
  layers_.push_back(new SimpleLayer(hidden_num_, output_num_, base_learning_rate_, 0.03));
}

void Net::LoadData() {
  ifstream input("/home/lwd/data/mnist/train-images.idx3-ubyte");
  unsigned char a;
  for(int i = 0; i < 16; ++i) {
    input >> noskipws >> a;
  }
  for(int i = 0; i < 60000; ++i) {
    input_[i].resize(784);
    for(int j = 0; j < 784; ++j) {
      input >> noskipws >> a;
      input_[i][j] = (int)a / 255.0 * 0.99 + 0.01;
    }
  }
  input.close();
  // label
  ifstream label("/home/lwd/data/mnist/train-labels.idx1-ubyte");
  for(int i = 0; i < 8; ++i) {
    label >> noskipws >> a;
  }
  for(int i = 0; i < 60000; ++i) {
    gt_[i].resize(10, 0.01);
    label >> noskipws >> a;
    gt_[i][int(a)] = 0.99;
  }
  label.close();
}

void Net::Forward(vector<float> in) {
  // std::cout << "*******Forward*******\n";
  layers_[0]->Forward(in);
  int i;
  for(i = 1; i < layer_num_; ++i) {
    layers_[i]->Forward(layers_[i-1]->output_);
  }
  output_.resize(layers_[i-1]->output_.size());
  for(int j = 0; j < output_.size(); ++j) {
    output_[j] = layers_[i-1]->output_[j];
    // std::cout << output_[j] << endl;
  }
}

void Net::Backward() {
  delta_.resize(output_num_);
  for(int i = 0; i < output_num_; ++i) {
    delta_[i] = (output_[i] - gt_[ran_num_][i])*output_[i]*(1-output_[i]);
  }
  layers_[layer_num_-1]->BackWard(delta_);
  int j;
  for(j = layer_num_- 2; j >= 0; --j) {
    layers_[j]->BackWard(layers_[j+1]->delta_);
  }
}

void Net::Train() {
  LoadData();
  srand((unsigned)time(NULL));
  int loop = 0;
  while(loop < 60000) {
    ran_num_ = rand() % 60000;
    ran_num_ = loop;
    Forward(input_[ran_num_]);
    float loss = 0;
    for(int i = 0; i < output_num_; ++i) {
      loss += (gt_[ran_num_][i]-output_[i])*(gt_[ran_num_][i]-output_[i]);
    }
    cout << "loop " << loop << " : loss " << loss << endl;
    if(loss < 0.0001) {
      break;
    }
    Backward();
    loop++;
  }
}

void Net::Test() {
  ifstream input("/home/lwd/data/mnist/t10k-images.idx3-ubyte");
  ifstream label("/home/lwd/data/mnist/t10k-labels.idx1-ubyte");
  unsigned char a;
  int cnt = 0;
  vector<float> tmp(784);
  for(int i = 0; i < 16; ++i) {
    input >> noskipws >> a;
  }
  for(int i = 0; i < 8; ++i) {
    label >> noskipws >> a;
  }
  for(int k = 0; k < 10000; ++k) {
    for(int i = 0; i < 784; ++i) {
      input >> noskipws >> a;
      tmp[i] = (int)a / 255.0 * 0.99 + 0.01;
    }
    Forward(tmp);
    float p = layers_[layer_num_-1]->output_[0];
    int index = 0;
    for(int i = 1; i < 10; ++i) {
      if(p < layers_[layer_num_-1]->output_[i]) {
        p = layers_[layer_num_-1]->output_[i];
        index = i;
      }
    }
    label >> noskipws >> a;
    cout << (int)a << "  " << index << endl;
    if(a == index) cnt++;
  }
  cout << cnt << endl;
  input.close();
  label.close();
}
#endif
