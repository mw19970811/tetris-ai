#pragma once
#include <array>
#include <vector>
#include "piece_data.h"
#include "tetris_core.h"

namespace tetris {

// Produces the identical 53-dim feature vector as Python's StateEncoder.
//
// Feature layout:
//   [0]     aggregate height (sum of column heights)
//   [1]     lines cleared placeholder (0.0)
//   [2]     hole count
//   [3]     bumpiness (sum of absolute differences between adjacent columns)
//   [4]     max well depth
//   [5]     mean height change placeholder (0.0)
//   [6:13]  current piece one-hot (7)
//   [13:17] current rotation one-hot (4)
//   [17:25] hold piece one-hot (8 = 7 types + empty)
//   [25:53] next queue one-hot (4 * 7 = 28)
class StateEncoder {
public:
    static constexpr int FEATURE_DIM = 53;

    StateEncoder(int cols = COLS, int total_rows = TOTAL_ROWS,
                 int hidden_rows = HIDDEN_ROWS, int next_queue_size = 4)
        : cols_(cols), total_rows_(total_rows), hidden_rows_(hidden_rows),
          next_queue_size_(next_queue_size) {}

    int featureDim() const { return FEATURE_DIM; }

    std::array<float, FEATURE_DIM> encodeFeatures(
        const Board& board,
        PieceName current_piece, int current_rotation,
        PieceName hold_piece, const std::vector<PieceName>& next_queue) const
    {
        std::array<float, FEATURE_DIM> feats{};

        // Use Board's cached column heights (O(1) after first call).
        const auto& heights = board.columnHeights();
        int total_h = 0;
        for (int c = 0; c < cols_; c++) total_h += heights[c];

        feats[0] = static_cast<float>(total_h);            // aggregate height
        feats[1] = 0.0f;                                    // lines cleared placeholder
        feats[2] = static_cast<float>(board.countHoles()); // holes
        feats[3] = static_cast<float>(board.bumpiness());  // bumpiness (uses cached heights)
        feats[4] = static_cast<float>(board.maxWellDepth()); // max well depth
        feats[5] = 0.0f;                                    // mean height change placeholder

        // Current piece one-hot (offset 6, 7 dims).
        int idx = 6;
        int pi = static_cast<int>(current_piece);
        if (pi >= 0 && pi < 7) feats[idx + pi] = 1.0f;

        // Rotation one-hot (offset 13, 4 dims).
        idx = 13;
        int ri = current_rotation % 4;
        if (ri >= 0 && ri < 4) feats[idx + ri] = 1.0f;

        // Hold piece one-hot (offset 17, 8 dims: 7 types + empty at index 7).
        idx = 17;
        int hi = static_cast<int>(hold_piece);
        if (hi < 0 || hi >= 7 || hold_piece == PieceName::NONE) {
            feats[idx + 7] = 1.0f;  // empty slot
        } else {
            feats[idx + hi] = 1.0f;
        }

        // Next queue one-hots (offset 25, each 7 dims).
        idx = 25;
        for (int i = 0; i < next_queue_size_; i++) {
            if (i < static_cast<int>(next_queue.size())) {
                int qi = static_cast<int>(next_queue[i]);
                if (qi >= 0 && qi < 7) {
                    feats[idx] = 1.0f;
                }
            }
            idx += 7;
        }

        return feats;
    }

private:
    int cols_;
    int total_rows_;
    int hidden_rows_;
    int next_queue_size_;
};

} // namespace tetris
