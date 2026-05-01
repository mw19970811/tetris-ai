#include "model_loader.h"
#include <cstring>
#include <stdexcept>

namespace tetris::inference {

ONNXModel::ONNXModel(const std::string& model_path, bool use_gpu)
    : env_(ORT_LOGGING_LEVEL_WARNING, "TetrisAI")
{
    if (use_gpu) {
        OrtCUDAProviderOptions cuda_opts;
        session_opts_.AppendExecutionProvider_CUDA(cuda_opts);
    }
    session_opts_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_opts_.SetIntraOpNumThreads(4);

    session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_opts_);

    // Get input/output names.
    size_t num_inputs = session_->GetInputCount();
    for (size_t i = 0; i < num_inputs; i++) {
        auto name = session_->GetInputNameAllocated(i, allocator_);
        input_names_.push_back(name.release());
    }
    size_t num_outputs = session_->GetOutputCount();
    for (size_t i = 0; i < num_outputs; i++) {
        auto name = session_->GetOutputNameAllocated(i, allocator_);
        output_names_.push_back(name.release());
    }

    // Get output shape to determine num_actions.
    auto type_info = session_->GetOutputTypeInfo(0);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    auto shape = tensor_info.GetShape();
    if (shape.size() >= 2) {
        num_actions_ = static_cast<int>(shape[1]);
    }

    board_shape_ = {1, 1, 22, 10};
    features_shape_ = {1, 53};
}

ONNXModel::~ONNXModel() {
    for (auto* name : input_names_) free(const_cast<char*>(name));
    for (auto* name : output_names_) free(const_cast<char*>(name));
}

void ONNXModel::infer(const float* board_data, const float* features_data,
                       float* q_values, int batch_size) {
    board_shape_[0] = batch_size;
    features_shape_[0] = batch_size;

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<Ort::Value> inputs;
    inputs.reserve(2);
    inputs.emplace_back(Ort::Value::CreateTensor<float>(
        memory_info, const_cast<float*>(board_data),
        batch_size * 1 * 22 * 10, board_shape_.data(), board_shape_.size()));
    inputs.emplace_back(Ort::Value::CreateTensor<float>(
        memory_info, const_cast<float*>(features_data),
        batch_size * 53, features_shape_.data(), features_shape_.size()));

    auto outputs = session_->Run(Ort::RunOptions{nullptr},
                                  input_names_.data(), inputs.data(), inputs.size(),
                                  output_names_.data(), output_names_.size());

    // Copy output.
    float* output_data = outputs[0].GetTensorMutableData<float>();
    size_t output_size = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
    std::memcpy(q_values, output_data, output_size * sizeof(float));
}

} // namespace tetris::inference
