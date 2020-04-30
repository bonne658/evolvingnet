#ifndef EVOLV_LAYER_H
#define EVOLV_LAYER_H
#include <vector>

using namespace std;

class Layer {
public:
  Layer(){}
  virtual void Forward(vector<float> input) = 0;
  virtual void BackWard(vector<float> delta) = 0;
  
  int iw_;
  int ih_;
  int ic_;
  int ow_;
  int oh_;
  int oc_;
  float lr_;
  vector<float> weight_;
  vector<float> input_;
  vector<float> output_;
  vector<float> bias_;
  vector<float> delta_;
};
#endif
