#pragma once
#include <string>
#include <vector>
#include <memory>
#include <cstdint>
#include <algorithm>
#include "model_loader.h"

// Forward declarations from env/core.
#include "../../env/core/piece_data.h"
#include "../../env/core/tetris_core.h"
#include "../../env/core/action_gen.h"
#include "../../env/core/tetris_env.h"

namespace tetris::inference {

class AIPlayer {
public:
    AIPlayer(const std::string& model_path, bool use_gpu = false)
        : model_(std::make_unique<ONNXModel>(model_path, use_gpu))
    {}

    // Select action for the current game state.
    Action selectAction(const TetrisEnv& env) {
        auto legal_actions = env.getLegalActions();
        if (legal_actions.empty()) {
            return Action{0, 0, false};
        }

        // Encode state.
        encodeState(env);

        // Run inference.
        std::vector<float> q_values(model_->num_actions(), 0.0f);
        model_->infer(board_input_.data(), features_input_.data(), q_values.data());

        // Select best legal action.
        return selectBest(q_values, legal_actions);
    }

private:
    void encodeState(const TetrisEnv& env) {
        const auto& state = env.getState();
        const auto& board = env.getBoard();

        // Board encoding: (1, 22, 10) float.
        board_input_.resize(1 * 22 * 10);
        for (int r = 0; r < TOTAL_ROWS; r++) {
            for (int c = 0; c < COLS; c++) {
                board_input_[r * COLS + c] = board.get(c, r) ? 1.0f : 0.0f;
            }
        }

        // Feature encoding (simplified — full version would match Python StateEncoder).
        features_input_.resize(53, 0.0f);
        // Column heights.
        const auto& heights = board.columnHeights();
        float height_sum = 0.0f;
        for (int c = 0; c < COLS; c++) height_sum += heights[c];
        features_input_[0] = height_sum;          // aggregate height
        features_input_[2] = board.countHoles();  // holes
        features_input_[3] = board.bumpiness();   // bumpiness
        features_input_[4] = board.maxWellDepth(); // max well

        // Current piece one-hot.
        int piece_idx = static_cast<int>(state.current_piece);
        features_input_[6 + piece_idx] = 1.0f;

        // Rotation one-hot.
        features_input_[13 + state.current_rotation] = 1.0f;

        // Hold one-hot.
        int hold_idx = static_cast<int>(state.hold_piece);
        if (hold_idx >= 0 && hold_idx < 7)
            features_input_[17 + hold_idx] = 1.0f;
        else
            features_input_[24] = 1.0f;  // empty

        // Next queue one-hots.
        for (size_t i = 0; i < state.next_queue.size() && i < 4; i++) {
            int n = static_cast<int>(state.next_queue[i]);
            if (n >= 0 && n < 7)
                features_input_[25 + i * 7 + n] = 1.0f;
        }
    }

    Action selectBest(const std::vector<float>& q_values,
                      const std::vector<Action>& legal_actions) {
        float best_q = -1e9f;
        Action best_action = legal_actions.empty() ? Action{0, 0, false} : legal_actions[0];

        for (const auto& action : legal_actions) {
            // Encode action to index.
            int col_idx = action.column + 2;
            col_idx = std::max(0, std::min(13, col_idx));
            int idx = action.rotation * 14 + col_idx;
            if (action.hold_first) idx += 56;
            idx = std::min(idx, model_->num_actions() - 1);

            if (q_values[idx] > best_q) {
                best_q = q_values[idx];
                best_action = action;
            }
        }
        return best_action;
    }

    std::unique_ptr<ONNXModel> model_;
    std::vector<float> board_input_;
    std::vector<float> features_input_;
};

} // namespace tetris::inference
