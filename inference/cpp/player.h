#pragma once
#include <string>
#include <vector>
#include <memory>
#include <cstdint>
#include <algorithm>
#include "model_loader.h"

#include "../../env/core/piece_data.h"
#include "../../env/core/tetris_core.h"
#include "../../env/core/action_gen.h"
#include "../../env/core/state_encoder.h"

namespace tetris::inference {

// One-stop AI Player: raw game state → best action.
// Internally handles feature encoding (StateEncoder), legal action
// generation (ActionGenerator), and ONNX inference (ONNXModel).
class AIPlayer {
public:
    AIPlayer(const std::string& model_path, bool use_gpu = false);
    ~AIPlayer() = default;

    // Select the best action for the given board state.
    Action selectAction(const Board& board,
                        PieceName current_piece, int current_rotation,
                        PieceName hold_piece, bool can_hold,
                        const std::vector<PieceName>& next_queue);

    // Run inference only — returns raw Q-values (model_->num_actions() floats).
    // Caller owns q_values buffer.
    void infer(const Board& board,
               PieceName current_piece, int current_rotation,
               PieceName hold_piece,
               const std::vector<PieceName>& next_queue,
               float* q_values);

    int numActions() const { return model_->num_actions(); }

private:
    // Encode board bitboard → (1, 22, 10) float array.
    void encodeBoard(const Board& board, float* out);

    // Select best legal action from Q-values (14-col bucket encoding).
    Action selectBest(const float* q_values,
                      const std::vector<Action>& legal_actions);

    std::unique_ptr<ONNXModel> model_;
    StateEncoder encoder_;
    std::vector<float> board_buf_;     // pre-alloc: 1 * TOTAL_ROWS * COLS
    std::vector<float> features_buf_;  // pre-alloc: 53
};

} // namespace tetris::inference
