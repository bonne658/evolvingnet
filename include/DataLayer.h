#ifndef EVOLV_DATA_LAYER_H
#define EVOLV_DATA_LAYER_H
#include "Layer.h"

using namespace std;

class DataLayer : public Layer {
public:
  DataLayer(int b, int w, int h, int c);
  shared_ptr<float> output;
};
#endif
