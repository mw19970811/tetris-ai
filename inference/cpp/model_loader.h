#pragma once
#include <string>
#include <vector>
#include <memory>
#include <onnxruntime_cxx_api.h>

namespace tetris::inference {

class ONNXModel {
public:
    ONNXModel(const std::string& model_path, bool use_gpu = false);
    ~ONNXModel();

    // Run inference. board_data = (1,1,22,10) float, features_data = (1,53) float.
    // q_values = (1, num_actions) float output.
    void infer(const float* board_data, const float* features_data,
               float* q_values, int batch_size = 1);

    int num_actions() const { return num_actions_; }

private:
    Ort::Env env_;
    Ort::SessionOptions session_opts_;
    std::unique_ptr<Ort::Session> session_;
    Ort::AllocatorWithDefaultOptions allocator_;

    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    std::vector<int64_t> board_shape_;
    std::vector<int64_t> features_shape_;
    int num_actions_ = 0;
};

} // namespace tetris::inference
