#ifdef USE_CUDNN
#include <vector>

#include "caffe/layers/cudnn_lcn_layer.hpp"

namespace caffe {

template <typename Dtype>
void CuDNNLCNLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  LRNLayer<Dtype>::LayerSetUp(bottom, top);

  CUDNN_CHECK(hipdnnCreate(&handle_));
  CUDNN_CHECK(hipdnnCreateLRNDescriptor(&norm_desc_));
  cudnn::createTensorDesc<Dtype>(&bottom_desc_);
  cudnn::createTensorDesc<Dtype>(&top_desc_);

  // create a LRN handle
  handles_setup_ = true;

  size_ = this->layer_param().lrn_param().local_size();
  pre_pad_ = (size_ - 1) / 2;
  alpha_ = this->layer_param().lrn_param().alpha();
  beta_ = this->layer_param().lrn_param().beta();
  k_ = this->layer_param().lrn_param().k();
}

template <typename Dtype>
void CuDNNLCNLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  LRNLayer<Dtype>::Reshape(bottom, top);
  cudnn::setTensorNdDesc<Dtype>(&bottom_desc_, bottom[0]->shape());
  cudnn::setTensorNdDesc<Dtype>(&top_desc_, top[0]->shape());
  CUDNN_CHECK(hipdnnSetLRNDescriptor(norm_desc_, size_, alpha_, beta_, k_));

  // allocate / reallocate tempData buffers
  size_t totalSizeInBytes = sizeof(Dtype)*bottom[0]->count();

  if (totalSizeInBytes > tempDataSize) {
    tempDataSize = totalSizeInBytes;

    hipFree(tempData1);
    hipFree(tempData2);

    // allocate new buffers
    CUDA_CHECK(hipMalloc(&tempData1, totalSizeInBytes));
    CUDA_CHECK(hipMalloc(&tempData2, totalSizeInBytes));
  }
}

template <typename Dtype>
CuDNNLCNLayer<Dtype>::~CuDNNLCNLayer() {
  // Check that handles have been setup before destroying.
  if (!handles_setup_) { return; }

  hipdnnDestroyTensorDescriptor(bottom_desc_);
  hipdnnDestroyTensorDescriptor(top_desc_);

  // destroy LRN handle
  hipdnnDestroy(handle_);

  // free temp buffers
  hipFree(tempData1);
  hipFree(tempData2);
}

INSTANTIATE_CLASS(CuDNNLCNLayer);

}   // namespace caffe
#endif
