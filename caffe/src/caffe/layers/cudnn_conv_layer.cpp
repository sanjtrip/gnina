#ifdef USE_CUDNN
#include <algorithm>
#include <vector>

#include "caffe/layers/cudnn_conv_layer.hpp"


namespace caffe {

// Set to three for the benefit of the backward pass, which
// can use separate streams for calculating the gradient w.r.t.
// bias, filter weights, and bottom data for each group independently
#define CUDNN_STREAMS_PER_GROUP 3

/**
 * TODO(dox) explain cuDNN interface
 */
template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  ConvolutionLayer<Dtype>::LayerSetUp(bottom, top);
  ConvolutionParameter conv_param = this->layer_param_.convolution_param();

  // Initialize CUDA streams and cuDNN.
  stream_         = new hipStream_t[this->group_ * CUDNN_STREAMS_PER_GROUP];
  handle_         = new hipdnnHandle_t[this->group_ * CUDNN_STREAMS_PER_GROUP];

  // Initialize algorithm arrays
  fwd_algo_       = new hipdnnConvolutionFwdAlgo_t[bottom.size()];
  bwd_filter_algo_= new hipdnnConvolutionBwdFilterAlgo_t[bottom.size()];
  bwd_data_algo_  = new hipdnnConvolutionBwdDataAlgo_t[bottom.size()];

  // initialize size arrays
  workspace_fwd_sizes_ = new size_t[bottom.size()];
  workspace_bwd_filter_sizes_ = new size_t[bottom.size()];
  workspace_bwd_data_sizes_ = new size_t[bottom.size()];

  // workspace data
  workspaceSizeInBytes = 0;
  workspaceData = NULL;
  workspace = new void*[this->group_ * CUDNN_STREAMS_PER_GROUP];

  for (size_t i = 0; i < bottom.size(); ++i) {
    // initialize all to default algorithms
    fwd_algo_[i] = (hipdnnConvolutionFwdAlgo_t)conv_param.cudnnconvolutionfwdalgo();
    bwd_filter_algo_[i] = (hipdnnConvolutionBwdFilterAlgo_t)conv_param.cudnnconvolutionbwdfilteralgo();
    bwd_data_algo_[i] = (hipdnnConvolutionBwdDataAlgo_t)conv_param.cudnnconvolutionbwddataalgo();
    // default algorithms don't require workspace
    workspace_fwd_sizes_[i] = 0;
    workspace_bwd_data_sizes_[i] = 0;
    workspace_bwd_filter_sizes_[i] = 0;
  }

  for (int g = 0; g < this->group_ * CUDNN_STREAMS_PER_GROUP; g++) {
    CUDA_CHECK(hipStreamCreate(&stream_[g]));
    CUDNN_CHECK(hipdnnCreate(&handle_[g]));
    CUDNN_CHECK(hipdnnSetStream(handle_[g], stream_[g]));
    workspace[g] = NULL;
  }

  // Set the indexing parameters.
  bias_offset_ = (this->num_output_ / this->group_);

  // Create filter descriptor.
  vector<int> kshape;
  kshape.push_back(this->num_output_ / this->group_);
  kshape.push_back( this->channels_ / this->group_);
  CHECK_EQ(this->kernel_shape_.shape().size(), 1) << "Unexpected kernel shape";
  kshape.insert(kshape.end(), this->kernel_shape_.cpu_data(), this->kernel_shape_.cpu_data()+this->kernel_shape_.count());
  cudnn::createFilterDesc<Dtype>(&filter_desc_, kshape);

  // Create tensor descriptor(s) for data and corresponding convolution(s).
  for (int i = 0; i < bottom.size(); i++) {
    hipdnnTensorDescriptor_t bottom_desc;
    cudnn::createTensorDesc<Dtype>(&bottom_desc);
    bottom_descs_.push_back(bottom_desc);
    hipdnnTensorDescriptor_t top_desc;
    cudnn::createTensorDesc<Dtype>(&top_desc);
    top_descs_.push_back(top_desc);
    hipdnnConvolutionDescriptor_t conv_desc;
    cudnn::createConvolutionDesc<Dtype>(&conv_desc);
    conv_descs_.push_back(conv_desc);
  }

  // Tensor descriptor for bias.
  if (this->bias_term_) {
    cudnn::createTensorDesc<Dtype>(&bias_desc_);
  }

  handles_setup_ = true;
}

//figure out what algorithm is fastest while still being deterministic and not taking up too much space
//leaving this code in here to evaluate other algorithms, but expectation is they will be configured manually
hipdnnConvolutionBwdFilterAlgo_t bestBackwardFilterAlgorithm(hipdnnHandle_t handle,
    const hipdnnTensorDescriptor_t          xDesc,
    const hipdnnTensorDescriptor_t          dyDesc,
    const hipdnnConvolutionDescriptor_t     convDesc,
    const hipdnnFilterDescriptor_t          dwDesc)
{
  int nalgo = 0, n = 0;
  size_t sz = 0;

  //the following code is TOO SLOW to be called from reshape
  CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(handle, &nalgo));
  hipdnnConvolutionBwdFilterAlgoPerf_t perfResults[nalgo];

  CUDNN_CHECK(hipdnnFindConvolutionBackwardFilterAlgorithm(handle, xDesc, dyDesc, convDesc,
      dwDesc, nalgo, &n, perfResults));

  //use size of data as guide for acceptable workspace size
  CUDNN_CHECK(cudnnGetTensorSizeInBytes(dyDesc, &sz));

  for(unsigned i = 0; i < n; i++) {
    //std::cout << "BFILT " << i << " " << perfResults[i].algo << " " << perfResults[i].determinism << " " << perfResults[i].time << " " << perfResults[i].memory << " " << perfResults[i].status << "\n";
    if(perfResults[i].status == HIPDNN_STATUS_SUCCESS && perfResults[i].determinism == CUDNN_DETERMINISTIC && perfResults[i].memory < sz) {
      return perfResults[i].algo;
    }
  }
  return HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT; //error
}

hipdnnConvolutionBwdDataAlgo_t bestBackwardDataAlgorithm(hipdnnHandle_t handle,
    const hipdnnFilterDescriptor_t          wDesc,
    const hipdnnTensorDescriptor_t          dyDesc,
    const hipdnnConvolutionDescriptor_t     convDesc,
    const hipdnnTensorDescriptor_t          dxDesc)
{
  int nalgo = 0, n = 0;
  size_t sz = 0;

  //the following code is TOO SLOW to be called from reshape
  CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithmMaxCount(handle, &nalgo));
  hipdnnConvolutionBwdDataAlgoPerf_t perfResults[nalgo];

  CUDNN_CHECK(hipdnnFindConvolutionBackwardDataAlgorithm(handle, wDesc, dyDesc, convDesc,
      dxDesc, nalgo, &n, perfResults));

  CUDNN_CHECK(cudnnGetTensorSizeInBytes(dyDesc, &sz));

  for(unsigned i = 0; i < n; i++) {
    //std::cout << "BDATA " << i << " " << perfResults[i].algo << " " << perfResults[i].determinism << " " << perfResults[i].time << " " << perfResults[i].memory << " " << perfResults[i].status << "\n";
    if(perfResults[i].status == HIPDNN_STATUS_SUCCESS && perfResults[i].determinism == CUDNN_DETERMINISTIC && perfResults[i].memory < sz) {
      return perfResults[i].algo;
    }
  }
  return HIPDNN_CONVOLUTION_BWD_DATA_ALGO_TRANSPOSE_GEMM; //error
}

hipdnnConvolutionFwdAlgo_t bestForwardAlgorithm(hipdnnHandle_t handle,
        const hipdnnTensorDescriptor_t      xDesc,
        const hipdnnFilterDescriptor_t      wDesc,
        const hipdnnConvolutionDescriptor_t convDesc,
        const hipdnnTensorDescriptor_t      yDesc)
{
  int nalgo = 0, n = 0;
  size_t sz = 0;

  //the following code is TOO SLOW to be called from reshape
  CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithmMaxCount(handle, &nalgo));
  hipdnnConvolutionFwdAlgoPerf_t perfResults[nalgo];

  CUDNN_CHECK(hipdnnFindConvolutionForwardAlgorithm(handle, xDesc, wDesc, convDesc,
      yDesc, nalgo, &n, perfResults));

  CUDNN_CHECK(cudnnGetTensorSizeInBytes(xDesc, &sz));

  for(unsigned i = 0; i < n; i++) {
    //std::cout << "FORW " << i << " " << perfResults[i].algo << " " << perfResults[i].determinism << " " << perfResults[i].time << " " << perfResults[i].memory << " " << perfResults[i].status << "\n";
    if(perfResults[i].status == HIPDNN_STATUS_SUCCESS && perfResults[i].determinism == CUDNN_DETERMINISTIC && perfResults[i].memory < sz) {
      return perfResults[i].algo;
    }
  }
  return HIPDNN_CONVOLUTION_FWD_ALGO_COUNT; //error
}

template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  ConvolutionLayer<Dtype>::Reshape(bottom, top);

  bottom_offset_ = this->bottom_dim_ / this->group_;
  top_offset_ = this->top_dim_ / this->group_;


  vector<int> pad; pad.assign(this->pad_.cpu_data(), this->pad_.cpu_data()+this->pad_.count());
  vector<int> stride; stride.assign(this->stride_.cpu_data(), this->stride_.cpu_data()+this->stride_.count());

  // Specify workspace limit for kernels directly until we have a
  // planning strategy and a rewrite of Caffe's GPU memory mangagement

  for (int i = 0; i < bottom.size(); i++) {
    cudnn::setTensorNdDesc<Dtype>(&bottom_descs_[i], bottom[i]->shape());
    cudnn::setTensorNdDesc<Dtype>(&top_descs_[i], top[i]->shape());

    cudnn::setConvolutionDesc<Dtype>(&conv_descs_[i], bottom_descs_[i],
        filter_desc_, pad, stride);

    // setup workspaces

    CUDNN_CHECK(hipdnnGetConvolutionForwardWorkspaceSize(handle_[0],
      bottom_descs_[i],
      filter_desc_,
      conv_descs_[i],
      top_descs_[i],
      fwd_algo_[i],
      &(workspace_fwd_sizes_[i])));

    // get workspace for backwards filter algorithm
    CUDNN_CHECK(hipdnnGetConvolutionBackwardFilterWorkspaceSize(handle_[0],
          bottom_descs_[i], top_descs_[i], conv_descs_[i], filter_desc_,
          bwd_filter_algo_[i], &workspace_bwd_filter_sizes_[i]));

    // get workspace size
    CUDNN_CHECK(hipdnnGetConvolutionBackwardDataWorkspaceSize(handle_[0],
          filter_desc_, top_descs_[i], conv_descs_[i], bottom_descs_[i],
          bwd_data_algo_[i], &workspace_bwd_data_sizes_[i]) );
  }

  // reduce over all workspace sizes to get a maximum to allocate / reallocate
  size_t total_workspace_fwd = 0;
  size_t total_workspace_bwd_data = 0;
  size_t total_workspace_bwd_filter = 0;

  for (size_t i = 0; i < bottom.size(); i++) {
    total_workspace_fwd        = std::max(total_workspace_fwd,
                                     workspace_fwd_sizes_[i]);
    total_workspace_bwd_data   = std::max(total_workspace_bwd_data,
                                     workspace_bwd_data_sizes_[i]);
    total_workspace_bwd_filter = std::max(total_workspace_bwd_filter,
                                     workspace_bwd_filter_sizes_[i]);
  }
  // get max over all operations
  size_t max_workspace = std::max(total_workspace_fwd,
                             total_workspace_bwd_data);
  max_workspace = std::max(max_workspace, total_workspace_bwd_filter);
  // ensure all groups have enough workspace
  size_t total_max_workspace = max_workspace *
                               (this->group_ * CUDNN_STREAMS_PER_GROUP);

  // this is the total amount of storage needed over all groups + streams
  if (total_max_workspace > workspaceSizeInBytes) {
    DLOG(INFO) << "Reallocating workspace storage: " << total_max_workspace;
    workspaceSizeInBytes = total_max_workspace;

    // free the existing workspace and allocate a new (larger) one
    hipFree(this->workspaceData);

    hipError_t err = hipMalloc(&(this->workspaceData), workspaceSizeInBytes);
    if (err != hipSuccess) {
      // force zero memory path
      for (int i = 0; i < bottom.size(); i++) {
        workspace_fwd_sizes_[i] = 0;
        workspace_bwd_filter_sizes_[i] = 0;
        workspace_bwd_data_sizes_[i] = 0;
        fwd_algo_[i] = HIPDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
        bwd_filter_algo_[i] = HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
        bwd_data_algo_[i] = HIPDNN_CONVOLUTION_BWD_DATA_ALGO_1;
      }

      // NULL out all workspace pointers
      for (int g = 0; g < (this->group_ * CUDNN_STREAMS_PER_GROUP); g++) {
        workspace[g] = NULL;
      }
      // NULL out underlying data
      workspaceData = NULL;
      workspaceSizeInBytes = 0;
    }

    // if we succeed in the allocation, set pointer aliases for workspaces
    for (int g = 0; g < (this->group_ * CUDNN_STREAMS_PER_GROUP); g++) {
      workspace[g] = reinterpret_cast<char *>(workspaceData) + g*max_workspace;
    }
  }

  // Tensor descriptor for bias.
  if (this->bias_term_) {
    std::vector<int> bias_shape;
    bias_shape.push_back(1);
    bias_shape.push_back(this->num_output_/this->group_);
    for(int i = 0; i < this->num_spatial_axes_; i++) {
      bias_shape.push_back(1);
    }
    cudnn::setTensorNdDesc<Dtype>(&bias_desc_, bias_shape);

  }
}

template <typename Dtype>
CuDNNConvolutionLayer<Dtype>::~CuDNNConvolutionLayer() {
  // Check that handles have been setup before destroying.
  if (!handles_setup_) { return; }

  for (int i = 0; i < bottom_descs_.size(); i++) {
    hipdnnDestroyTensorDescriptor(bottom_descs_[i]);
    hipdnnDestroyTensorDescriptor(top_descs_[i]);
    hipdnnDestroyConvolutionDescriptor(conv_descs_[i]);
  }
  if (this->bias_term_) {
    hipdnnDestroyTensorDescriptor(bias_desc_);
  }
  hipdnnDestroyFilterDescriptor(filter_desc_);

  for (int g = 0; g < this->group_ * CUDNN_STREAMS_PER_GROUP; g++) {
    hipStreamDestroy(stream_[g]);
    hipdnnDestroy(handle_[g]);
  }

  hipFree(workspaceData);
  delete [] workspace;
  delete [] stream_;
  delete [] handle_;
  delete [] fwd_algo_;
  delete [] bwd_filter_algo_;
  delete [] bwd_data_algo_;
  delete [] workspace_fwd_sizes_;
  delete [] workspace_bwd_data_sizes_;
  delete [] workspace_bwd_filter_sizes_;
}

INSTANTIATE_CLASS(CuDNNConvolutionLayer);

}   // namespace caffe
#endif
