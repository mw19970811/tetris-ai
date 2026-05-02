#include "player.h"
#include <cstring>

namespace tetris::inference {

AIPlayer::AIPlayer(const std::string& model_path, bool use_gpu)
    : model_(std::make_unique<ONNXModel>(model_path, use_gpu))
{
    board_buf_.resize(1 * TOTAL_ROWS * COLS);
    features_buf_.resize(StateEncoder::FEATURE_DIM);
}

Action AIPlayer::selectAction(const Board& board,
                               PieceName current_piece, int current_rotation,
                               PieceName hold_piece, bool can_hold,
                               const std::vector<PieceName>& next_queue)
{
    auto legal_actions = ActionGenerator::getLegalActions(
        board, current_piece, current_rotation, hold_piece, can_hold);

    if (legal_actions.empty()) {
        return Action{0, 0, false};
    }

    std::vector<float> q_values(static_cast<size_t>(model_->num_actions()), 0.0f);
    infer(board, current_piece, current_rotation, hold_piece,
          next_queue, q_values.data());

    return selectBest(q_values.data(), legal_actions);
}

void AIPlayer::infer(const Board& board,
                      PieceName current_piece, int current_rotation,
                      PieceName hold_piece,
                      const std::vector<PieceName>& next_queue,
                      float* q_values)
{
    encodeBoard(board, board_buf_.data());

    auto feats = encoder_.encodeFeatures(
        board, current_piece, current_rotation, hold_piece, next_queue);
    std::copy(feats.begin(), feats.end(), features_buf_.begin());

    model_->infer(board_buf_.data(), features_buf_.data(), q_values);
}

void AIPlayer::encodeBoard(const Board& board, float* out) {
    for (int r = 0; r < TOTAL_ROWS; r++) {
        uint16_t row = board.row(r);
        for (int c = 0; c < COLS; c++) {
            out[r * COLS + c] = (row >> c) & 1u ? 1.0f : 0.0f;
        }
    }
}

Action AIPlayer::selectBest(const float* q_values,
                             const std::vector<Action>& legal_actions)
{
    float best_q = -1e9f;
    Action best_action = legal_actions[0];

    for (const auto& action : legal_actions) {
        // 14-col bucket encoding (matches action_mask.py).
        int col_idx = action.column + 2;
        col_idx = std::max(0, std::min(13, col_idx));
        int idx = action.rotation * 14 + col_idx;
        if (action.hold) idx += 56;
        idx = std::min(idx, model_->num_actions() - 1);

        if (q_values[idx] > best_q) {
            best_q = q_values[idx];
            best_action = action;
        }
    }
    return best_action;
}

} // namespace tetris::inference
