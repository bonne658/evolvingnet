#include "Net.h"
#include "ConnectLayer.h"
#include "ConvolutionLayer.h"
#include "DataLayer.h"
#include "PoolingLayer.h"
#include "ReluLayer.h"

Net::Net() {
  layers_.push_back(new DataLayer(32, 32, 1));
  layers_.push_back(new ConvolutionLayer());
  layers_.push_back(new PoolingLayer());
  layers_.push_back(new ConvolutionLayer());
  layers_.push_back(new PoolingLayer());
  layers_.push_back(new ConnectLayer());
  layers_.push_back(new ReluLayer());
  layers_.push_back(new ConnectLayer());
}

void Net::Forward() {
  layers_[0]->Forward();
}
