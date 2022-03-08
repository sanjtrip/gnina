#include "precalculate_gpu.h"
#include "gpucode.h"
#include "gpu_util.h"

//evaluates splines at t1/t2 and r, properly swaping result
component_pair precalculate_gpu::evaldata(smt t1, smt t2, fl r) const {
  if (t1 <= t2) {
    return cpudata(t1, t2).eval(r);
  } else {
    component_pair ret = cpudata(t2, t1).eval(r);
    ret.first.swapOrder();
    ret.second.swapOrder();
    return ret;
  }

  //DEAD CODE below - using the gpu for this small calculation is very inefficient
  // std::vector<float> vals, derivs;
  // if(t1 <= t2){
  // evaluate_splines(data(t1, t2), r, vals, derivs);
  // return component_pair(result_components(vals), result_components(derivs));
  // } else{
  // evaluate_splines(data(t2, t1), r, vals, derivs);
  // component_pair ret = component_pair(result_components(vals),
  // result_components(derivs));
  // ret.first.swapOrder();
  // ret.second.swapOrder();
  // return ret;
  // }
}

void precalculate_gpu::evaluate_splines(const GPUSplineInfo& spInfo, float r,
    std::vector<float>& vals, std::vector<float>& derivs) const {
  unsigned n = spInfo.n;
  vals.resize(n);
  derivs.resize(n);

  evaluate_splines_host(spInfo, r, device_vals, device_derivs);

  hipMemcpy(&vals[0], device_vals, n * sizeof(float), hipMemcpyDeviceToHost);
  hipMemcpy(&derivs[0], device_derivs, n * sizeof(float),
      hipMemcpyDeviceToHost);
}

void precalculate_gpu::setup_points(std::vector<std::vector<pr> >& points,
    smt t1, smt t2, double cutoff, unsigned n,
    const scoring_function& sf) const {
  assert(n >= 2);
  fl fraction = cutoff / (fl) n;
  sz numc = sf.num_used_components();

  //clear out arguments
  points.resize(numc);
  for (sz i = 0; i < numc; i++) {
    points[i].clear();
    points[i].reserve(numc + 1);
  }

  //compute points
  for (unsigned i = 0; i < n; i++) {
    fl xval = i * fraction;
    result_components res = sf.eval_fast(t1, t2, xval);
    for (unsigned c = 0; c < numc; c++) {
      points[c].push_back(pr(xval, res[c]));
    }
  }
  //last point at cutoff is zero
  for (unsigned c = 0; c < numc; c++) {
    points[c].push_back(pr(cutoff, 0));
  }
}

precalculate_gpu::precalculate_gpu(const scoring_function& sf, fl factor_)
    :
        // sf should not be discontinuous, even near cutoff, for the sake of the derivatives
        precalculate(sf), data(num_atom_types(), GPUSplineInfo()),
        cpudata(num_atom_types(), spline_cache()), deviceData(NULL),
        device_vals(NULL), device_derivs(
        NULL), delta(0.000005), factor(factor_) {

  VINA_CHECK(factor > epsilon_fl);
  unsigned n = factor * m_cutoff;
  unsigned pitch = data.dim();
  unsigned datasize = pitch * (pitch + 1) / 2;
  GPUSplineInfo devicedata[datasize]; //todo, investigate using 2d array
  unsigned maxcomponents = 0;
//Prepare to store and then transfer all spline data to GPU
  std::vector<FloatSplineData> allSplineData;
  std::vector<int> indices;
  std::vector<void *> deviceMem;
  float **splines;
  int numc = sf.num_used_components();
  std::vector<int> locations;
  allSplineData.reserve(datasize * data.dim());

  if (sf.has_slow()) {
    std::cerr << "GPU acceleration does not support 'slow' scoring terms\n";
    abort();
  }
  VINA_FOR(t1, data.dim()) {

    VINA_RANGE(t2, t1, data.dim()) {

      cpudata(t1, t2).set(sf, (smt) t1, (smt) t2, m_cutoff, n);

      //create the splines and copy the data over to the gpu
      //todo: do spline interpolation on gpu
      Spline spline;
      //create spline
      std::vector<std::vector<pr> > points;
      setup_points(points, (smt) t1, (smt) t2, m_cutoff, n, sf);
      std::vector<void *> deviceMem;
      float fraction = 0, cutoff = 0;
      for (sz i = 0, n = points.size(); i < n; i++) {
        //the size of points is the number of components
        Spline spline;
        spline.initialize(points[i]);
        fraction = spline.getFraction();
        cutoff = spline.getCutoff();
        //convert to float
        const std::vector<SplineData>& data = spline.getData();
        locations.push_back((int) allSplineData.size());
        for (sz j = 0, m = data.size(); j < m; j++) {
          allSplineData.push_back(FloatSplineData(data[j]));
        }
      }

      GPUSplineInfo info;
      info.n = numc;
      info.fraction = fraction;
      info.cutoff = cutoff;
      data(t1, t2) = info;

      if (info.n > maxcomponents) maxcomponents = info.n;

      unsigned tindex = triangular_matrix_index_permissive(pitch, t1, t2);
      devicedata[tindex] = info;
      indices.push_back(tindex);
    }
  }
  //copy devicedata to gpu
  void *deviceAllSplineData = NULL;
  hipMalloc(&deviceAllSplineData,
      sizeof(FloatSplineData) * allSplineData.size());

  hipMemcpy(deviceAllSplineData, &allSplineData[0],
      sizeof(FloatSplineData) * allSplineData.size(), hipMemcpyHostToDevice);
  for (sz i = 0, j = locations.size(); i < j; i++) {
    deviceMem.push_back((FloatSplineData*) deviceAllSplineData + locations[i]);
  }

  hipMalloc(&splines, sizeof(FloatSplineData*) * deviceMem.size());
  hipMemcpy(splines, &deviceMem[0],
      sizeof(FloatSplineData*) * deviceMem.size(), hipMemcpyHostToDevice);

  assert(indices.size() == datasize);
  //finally, fill in splines pointer in GPUSplineInfo using points.size()*tindex
  for (sz m = 0, p = indices.size(); m < p; m++) {
    devicedata[indices[m]].splines = splines + m * numc;
  }

  //copy device data

  hipMalloc(&deviceData, sizeof(GPUSplineInfo) * datasize);
  hipMemcpy(deviceData, devicedata, sizeof(GPUSplineInfo) * datasize,
      hipMemcpyHostToDevice);

  //allocate buffers for spline host-device computation
  hipMalloc(&device_vals, sizeof(float) * maxcomponents);
  hipMalloc(&device_derivs, sizeof(float) * maxcomponents);

}

precalculate_gpu::~precalculate_gpu() {
  //need to cuda free splines
  if (deviceData) hipFree(deviceData);

  if (device_vals) hipFree(device_vals);
  if (device_derivs) hipFree(device_derivs);

  unsigned n = data.dim();
  for (unsigned i = 0; i < n; i++) {
    for (unsigned j = i; j < n; j++) {
      //free each spline, which means we have to get the pointers
      const GPUSplineInfo& info = data(i, j);
      if (info.splines) {
        void *splines[info.n];
        hipMemcpy(splines, info.splines, sizeof(float*) * info.n,
            hipMemcpyDeviceToHost);
        for (unsigned k = 0; k < info.n; k++) {
          if (splines[k]) hipFree(splines[k]);
        }
        hipFree(info.splines);
      }
    }
  }
}

result_components precalculate_gpu::eval_fast(smt t1, smt t2, fl r2) const {
  assert(r2 <= m_cutoff_sqr);
  fl r = sqrt(r2);
  return evaldata(t1, t2, r).first;
}

pr precalculate_gpu::eval_deriv(const atom_base& a, const atom_base& b,
    fl r2) const {
  assert(r2 <= m_cutoff_sqr);
  smt t1 = a.get();
  smt t2 = b.get();
  fl r = sqrt(r2);
  component_pair rets = evaldata(t1, t2, r);
  pr ret(rets.first.eval(a, b), rets.second.eval(a, b));

  if (scoring.has_slow()) {
    //compute value and numerical derivative directly from function
    fl X = scoring.eval_slow(a, b, r);
    ret.first += X;

    fl rhi = r + delta;
    fl rlo = r - delta;
    if (rlo < 0) rlo = 0;
    if (rhi > m_cutoff) rhi = m_cutoff;

    fl W = scoring.eval_slow(a, b, rlo);
    fl Y = 0;
    if (rhi < m_cutoff) Y = scoring.eval_slow(a, b, rhi);

    fl dx = (Y - W) / (rhi - rlo);
    ret.second += dx;
  }
  ret.second /= r;
  return ret;
}
