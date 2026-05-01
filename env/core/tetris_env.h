#pragma once
#include <cstdint>
#include <array>
#include <vector>
#include <string>
#include <memory>
#include <random>
#include "piece_data.h"
#include "tetris_core.h"
#include "randomizer.h"
#include "action_gen.h"

namespace tetris {

struct EnvConfig {
    int cols = COLS;
    int rows = ROWS;
    int hidden_rows = HIDDEN_ROWS;
    int lock_delay_ms = LOCK_DELAY_MS;
    int lock_moves_max = LOCK_MOVES_MAX;
    int next_queue_size = 4;
    std::string bag_type = "7bag";  // "7bag" or "random"

    // Reward weights.
    float w_height = 0.3f;
    float w_holes = 1.5f;
    float w_bumpiness = 0.2f;
    float w_well = 0.5f;
    float w_survival = 0.01f;
    float w_death = -100.0f;
    float hard_drop_score = 2.0f;
    float soft_drop_score = 1.0f;
};

struct EnvState {
    Board board;
    PieceName current_piece;
    int current_rotation;
    PieceName hold_piece;
    bool can_hold;
    std::vector<PieceName> next_queue;  // size = next_queue_size
    int score;
    int level;
    int lines_cleared;
    bool terminated;
    int step_count;
};

class TetrisEnv {
public:
    TetrisEnv(const EnvConfig& config = EnvConfig{}, uint64_t seed = 0)
        : config_(config), randomizer_(seed), rng_(seed)
    {
        if (seed == 0) {
            std::random_device rd;
            rng_.seed(rd());
            randomizer_.seed(rd());
        }
        reset();
    }

    void reset() {
        state_.board.clear();
        state_.current_piece = PieceName::NONE;
        state_.current_rotation = 0;
        state_.hold_piece = PieceName::NONE;
        state_.can_hold = true;
        state_.score = 0;
        state_.level = 1;
        state_.lines_cleared = 0;
        state_.terminated = false;
        state_.step_count = 0;

        randomizer_.reset();
        state_.next_queue.clear();
        for (int i = 0; i < config_.next_queue_size + 1; i++) {
            state_.next_queue.push_back(randomizer_.next());
        }

        spawnPiece();
    }

    // Execute a placement action. Returns the reward.
    float step(const Action& action) {
        if (state_.terminated) return 0.0f;

        float reward = 0.0f;
        PieceName placed_piece_name = state_.current_piece;

        // Handle hold.
        if (action.hold) {
            if (state_.can_hold) {
                if (state_.hold_piece == PieceName::NONE) {
                    // Hold current piece, spawn next.
                    state_.hold_piece = state_.current_piece;
                    state_.can_hold = false;
                    spawnPiece();
                    // Re-evaluate legal actions with new piece (the hold-and-skip case).
                    // For placement-based, hold-and-skip is just the hold action itself.
                    // We return small reward and continue.
                    return config_.w_survival;
                } else {
                    // Swap hold with current.
                    std::swap(state_.hold_piece, state_.current_piece);
                    state_.can_hold = false;
                    placed_piece_name = state_.current_piece;
                }
            }
        }

        // Place the piece at the target position.
        const auto& cells = PIECES[static_cast<int>(placed_piece_name)].rotations[action.rotation];
        int ghost_y = state_.board.ghostY(cells, action.column, HIDDEN_ROWS - 2);

        // Compute drop distance for hard drop bonus.
        int drop_dist = ghost_y - (HIDDEN_ROWS - 2);
        reward += drop_dist * config_.hard_drop_score;

        // Place piece.
        state_.board.placePiece(cells, action.column, ghost_y, placed_piece_name);

        // Check game over (blocks in hidden rows).
        if (state_.board.blocksInHiddenRows()) {
            state_.terminated = true;
            reward += config_.w_death;
            state_.score += static_cast<int>(reward);
            return reward;
        }

        // Clear lines and get score.
        int lines_cleared = state_.board.clearLines();
        if (lines_cleared > 0) {
            int line_score = SCORE_TABLE[lines_cleared] * state_.level;
            reward += static_cast<float>(line_score);
            state_.lines_cleared += lines_cleared;
            state_.level = (state_.lines_cleared / 10) + 1;
        }

        // Reward shaping: board quality penalties.
        reward -= config_.w_height * state_.board.totalHeight();
        reward -= config_.w_holes * state_.board.countHoles();
        reward -= config_.w_bumpiness * state_.board.bumpiness();
        reward -= config_.w_well * state_.board.maxWellDepth();
        reward += config_.w_survival;  // survival bonus.

        state_.score += static_cast<int>(reward);
        state_.step_count++;

        // Spawn next piece.
        spawnPiece();

        return reward;
    }

    // Get the current state data for feature encoding.
    const EnvState& getState() const { return state_; }
    const Board& getBoard() const { return state_.board; }

    // Get legal actions for current state.
    std::vector<Action> getLegalActions() const {
        return ActionGenerator::getLegalActions(
            state_.board,
            state_.current_piece, state_.current_rotation,
            state_.hold_piece, state_.can_hold
        );
    }

    // Get boolean action mask compatible with Python's 10-col encoding scheme.
    // Encoding: idx = rotation * 10 + (column + 2) + (hold ? 40 : 0)
    std::vector<bool> getLegalActionsMask(int max_actions = 112) const {
        std::vector<bool> mask(max_actions, false);
        auto actions = getLegalActions();
        for (const auto& a : actions) {
            int col_idx = a.column + 2;
            if (col_idx < 0) col_idx = 0;
            if (col_idx >= 10) col_idx = 9;
            int idx = a.rotation * 10 + col_idx;
            if (a.hold) idx += 40;
            if (idx >= 0 && idx < max_actions) mask[idx] = true;
        }
        return mask;
    }

    // Get drop interval for current level (ms).
    int getDropInterval() const {
        int idx = std::min(state_.level - 1, static_cast<int>(sizeof(DROP_SPEEDS) / sizeof(DROP_SPEEDS[0]) - 1));
        idx = std::max(0, idx);
        return DROP_SPEEDS[idx];
    }

    bool isTerminated() const { return state_.terminated; }
    int getScore() const { return state_.score; }
    int getLevel() const { return state_.level; }
    int getLinesCleared() const { return state_.lines_cleared; }

    const EnvConfig& config() const { return config_; }

private:
    void spawnPiece() {
        // Shift next queue: pop front, append new from randomizer.
        if (!state_.next_queue.empty()) {
            state_.current_piece = state_.next_queue.front();
            state_.next_queue.erase(state_.next_queue.begin());
        }
        state_.next_queue.push_back(randomizer_.next());
        state_.current_rotation = 0;
        state_.can_hold = true;

        // Check if spawn position is valid.
        const auto& cells = PIECES[static_cast<int>(state_.current_piece)].rotations[0];
        if (state_.board.collides(cells, 3, HIDDEN_ROWS - 2)) {
            state_.terminated = true;
        }
    }

    EnvConfig config_;
    EnvState state_;
    BagRandomizer randomizer_;
    mutable std::mt19937_64 rng_;
};

} // namespace tetris
