#include "player.h"
#include <cstring>

namespace tetris::inference {

AIPlayer::AIPlayer(const std::string& model_path, bool use_gpu)
    : model_(std::make_unique<ONNXModel>(model_path, use_gpu))
{}

Action AIPlayer::selectAction(const TetrisEnv& env) {
    auto legal_actions = env.getLegalActions();
    if (legal_actions.empty()) {
        return Action{0, 0, false};
    }

    encodeState(env);

    std::vector<float> q_values(static_cast<size_t>(model_->num_actions()), 0.0f);
    model_->infer(board_input_.data(), features_input_.data(), q_values.data());

    return selectBest(q_values, legal_actions);
}

void AIPlayer::encodeState(const TetrisEnv& env) {
    const auto& state = env.getState();
    const auto& board = env.getBoard();

    board_input_.resize(1 * TOTAL_ROWS * COLS);
    for (int r = 0; r < TOTAL_ROWS; r++) {
        for (int c = 0; c < COLS; c++) {
            board_input_[r * COLS + c] = board.get(c, r) ? 1.0f : 0.0f;
        }
    }

    features_input_.assign(53, 0.0f);
    const auto& heights = board.columnHeights();
    float height_sum = 0.0f;
    for (int c = 0; c < COLS; c++) height_sum += static_cast<float>(heights[c]);

    features_input_[0] = height_sum;
    features_input_[2] = static_cast<float>(board.countHoles());
    features_input_[3] = static_cast<float>(board.bumpiness());
    features_input_[4] = static_cast<float>(board.maxWellDepth());

    int piece_idx = static_cast<int>(state.current_piece);
    if (piece_idx >= 0 && piece_idx < 7)
        features_input_[6 + piece_idx] = 1.0f;
    features_input_[13 + state.current_rotation] = 1.0f;

    int hold_idx = static_cast<int>(state.hold_piece);
    if (hold_idx >= 0 && hold_idx < 7)
        features_input_[17 + hold_idx] = 1.0f;
    else
        features_input_[24] = 1.0f;

    for (size_t i = 0; i < state.next_queue.size() && i < 4; i++) {
        int n = static_cast<int>(state.next_queue[i]);
        if (n >= 0 && n < 7)
            features_input_[25 + i * 7 + n] = 1.0f;
    }
}

Action AIPlayer::selectBest(const std::vector<float>& q_values,
                            const std::vector<Action>& legal_actions) {
    float best_q = -1e9f;
    Action best_action = legal_actions.empty() ? Action{0, 0, false} : legal_actions[0];

    for (const auto& action : legal_actions) {
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

} // namespace tetris::inference
