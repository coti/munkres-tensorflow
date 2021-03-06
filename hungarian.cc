#include <assert.h>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/util/work_sharder.h"
#include "tensorflow/core/util/tensor_format.h"
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
        
        string data_format;
        c->GetAttr("data_format", &data_format);
        int nodes_axis = (data_format == "NHWC") ? 1 : 2;

        if (!c->RankKnown(input)) {
          // If we do not have the rank of the input, we don't know the output shape.
          c->set_output(0, c->UnknownShape());
          return Status::OK();
        }

        const int32 input_rank = c->Rank(input);
        std::vector< DimensionHandle> dims;
        for (int i = 0; i < input_rank; ++i) {
            dims.emplace_back(c->Dim(input, i));
            if (i == nodes_axis) {
                    dims.emplace_back(c->Dim(input, 2));
            }
        }
        c->set_output(0, c->MakeShape(dims));

        return Status::OK();
    });

//TODO: add error handling for inf or input with overflow
class HungarianOp : public OpKernel {
 public:
      explicit HungarianOp(OpKernelConstruction* context) : OpKernel(context) {
          string data_format;
          if (context->GetAttr("data_format", &data_format).ok()) {
              OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                          errors::InvalidArgument("Invalid data format"));
          } else {
              data_format_ = FORMAT_NHWC;
          }
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& costs_tensor = context->input(0);
    auto costs = costs_tensor.tensor<float, 4>();
    const Tensor& adj = context->input(1);
    auto adjacency = adj.tensor<bool, 3>();
    // Create an output tensor
    Tensor* assignments_tensor = NULL;
      
    const TensorShape& cost_shape = costs_tensor.shape();
    TensorShape output_shape = cost_shape;
    output_shape.InsertDim((data_format_ == FORMAT_NHWC) ? 1 : 2, cost_shape.dim_size(2));
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape,
                                                     &assignments_tensor));
    
    auto input_shape = cost_shape.dim_sizes();
    auto output_flat = assignments_tensor->flat<bool>();

    // Set all but the first element of the output tensor to 0.
    const int N = output_flat.size();
    for (int i = 1; i < N; i++) {
      output_flat(i) = false;
    }

    auto assignments_output = assignments_tensor->tensor<bool, 5>();

    const int batch_size = input_shape[0]*input_shape[1]*input_shape[2]; // [gfkia]
    TensorFormat data_format = data_format_;
    std::function<void(int64, int64)> shard; 
    if (data_format == FORMAT_NHWC) {
        shard = [&costs, &input_shape, &assignments_output, &adjacency](int64 start, int64 limit) {
            for (int job = start; job < limit; ++job) {
                int graph = job / input_shape[1] / input_shape[2];
                int n = job / input_shape[2] % input_shape[1];
                int node = job % input_shape[2];
                vector<int> ids;
                for (int k = 0; k < input_shape[2]; ++k) {
                    if(adjacency(graph, node, k)) {
                        ids.push_back(k);
                    }
                }
                if (ids.size() > 0) {
                    Matrix<float> matrix(ids.size(), input_shape[3]);
                    for (int i = 0; i < ids.size(); ++i) {
                        for (int j = 0; j < input_shape[3]; ++j) {
                            matrix(i,j) = costs(graph, n, ids[i], j);
                        }
                    }
                    Munkres<float> munk = Munkres<float>();
                    munk.solve(matrix);

                    for (int i = 0; i < ids.size(); ++i) {
                        for (int j = 0; j < input_shape[3]; ++j){
                            if(matrix(i,j) == 0){
                                    assignments_output(graph, node, n, ids[i], j) = true;
                            }
                        }
                    }
                }
            }
        };
    }
    else {
        shard = [&costs, &input_shape, &assignments_output, &adjacency](int64 start, int64 limit) {
            for (int job = start; job < limit; ++job) {
                int graph = job / input_shape[1] / input_shape[2];
                int n = job / input_shape[2] % input_shape[1];
                int node = job % input_shape[2];
                vector<int> ids;
                for (int k = 0; k < input_shape[2]; ++k) {
                    if(adjacency(graph, node, k)) {
                        ids.push_back(k);
                    }
                }
                if (ids.size() > 0) {
                    Matrix<float> matrix(ids.size(), input_shape[3]);
                    for (int i = 0; i < ids.size(); ++i) {
                        for (int j = 0; j < input_shape[3]; ++j) {
                            matrix(i,j) = costs(graph, n, ids[i], j);
                        }
                    }
                    Munkres<float> munk = Munkres<float>();
                    munk.solve(matrix);

                    for (int i = 0; i < ids.size(); ++i) {
                        for (int j = 0; j < input_shape[3]; ++j){
                            if(matrix(i,j) == 0){
                                assignments_output(graph, n, node, ids[i], j) = true;
                            }
                        }
                    }
                }
            }
        };
    }

    // This is just a very crude approximation
    const int64 single_cost = 10000 * input_shape[2] * input_shape[3];

    auto worker_threads = context->device()->tensorflow_cpu_worker_threads();
    Shard(worker_threads->num_threads, worker_threads->workers, batch_size, single_cost, shard);
  }

 private:
    TensorFormat data_format_;
};

REGISTER_KERNEL_BUILDER(Name("Hungarian").Device(DEVICE_CPU), HungarianOp);
