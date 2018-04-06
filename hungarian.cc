#include <assert.h>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/util/work_sharder.h"
#include <iostream>
#include "munkres.h"

using std::vector;
using namespace tensorflow;
using shape_inference::Shape;
using shape_inference::Dimension;
using shape_inference::DimensionHandle;
using shape_inference::ShapeHandle;

REGISTER_OP("Hungarian")
    .Input("cost_matrix: float32")
    .Input("adjacency: bool")
    .Output("assignments: bool")
    //TODO: add shape function
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      //Same shape as input, droping last dimension, as an index is returned for each assignment
        ShapeHandle input = c->input(0);
        ShapeHandle adj = c->input(1);

        if (!c->RankKnown(input)) {
          // If we do not have the rank of the input, we don't know the output shape.
          c->set_output(0, c->UnknownShape());
          return Status::OK();
        }

        const int32 input_rank = c->Rank(input);
        std::vector< DimensionHandle> dims;
        for (int i = 0; i < input_rank; ++i) {
            dims.emplace_back(c->Dim(input, i));
            if (i == 2) {
                    dims.emplace_back(c->Dim(input, i));
            }
        }
        c->set_output(0, c->MakeShape(dims));

        return Status::OK();
    });

//TODO: add error handling for inf or input with overflow
class HungarianOp : public OpKernel {
 public:
  explicit HungarianOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& costs_tensor = context->input(0);
    auto costs = costs_tensor.tensor<float, 4>();
    const Tensor& adj = context->input(1);
    auto adjacency = adj.tensor<bool, 3>();
    // Create an output tensor
    Tensor* assignments_tensor = NULL;
      
    const TensorShape& input_shape = costs_tensor.shape();
    TensorShape output_shape = input_shape;
    output_shape.InsertDim(2, input_shape.dim_size(2));
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape,
                                                     &assignments_tensor));
    
    auto cost_shape = output_shape.dim_sizes();
    auto output_flat = assignments_tensor->flat<bool>();

    // Set all but the first element of the output tensor to 0.
    const int N = output_flat.size();
    for (int i = 1; i < N; i++) {
      output_flat(i) = false;
    }

    auto assignments_output = assignments_tensor->tensor<bool, 5>();

    const int batch_size = cost_shape[0];
    auto shard = [&costs, &cost_shape, &assignments_output, &adjacency](int64 start, int64 limit) {
        for (int graph = start; graph < limit; ++graph) {
            for (int node = 0; node < cost_shape[2]; ++node) {
              vector<int> ids;
                  for (int k = 0; k < cost_shape[2]; ++k) {
                      if(adjacency(graph, node, k)) {
                          ids.push_back(k);
                      }
                  }
                for (int n = 0; n < cost_shape[1]; ++n) {
                    Matrix<float> matrix(ids.size(), cost_shape[4]);
                    for (int i = 0; i < ids.size(); ++i) {
                      for (int j = 0; j < cost_shape[4]; ++j) {
                        matrix(i,j) = costs(graph, n, ids[i], j);
                      }
                    }
                    Munkres<float> munk = Munkres<float>();
                    munk.solve(matrix);

                    for (int i = 0; i < ids.size(); ++i) {
                      for (int j = 0; j < cost_shape[4]; ++j){
                        if(matrix(i,j) == 0){
                          assignments_output(graph, n, node, ids[i], j) = true;
                        }
                      }
                    }
                }
            }
      }
    };

    // This is just a very crude approximation
    const int64 single_cost = 10000 * cost_shape[1] * cost_shape[1] * cost_shape[2] * cost_shape[3] * cost_shape[3];

    auto worker_threads = context->device()->tensorflow_cpu_worker_threads();
    Shard(worker_threads->num_threads, worker_threads->workers, batch_size, single_cost, shard);
  }
};

REGISTER_KERNEL_BUILDER(Name("Hungarian").Device(DEVICE_CPU), HungarianOp);
