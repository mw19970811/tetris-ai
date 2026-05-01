#pragma once
#include <cstdint>
#include <cstring>
#include <array>
#include <vector>
#include <algorithm>
#include "piece_data.h"

namespace tetris {

// Bitboard-based Tetris board.
// Each row is a uint16_t where bits 0-9 represent columns 0-9.
// Row 0 = top (hidden rows area), Row 21 = bottom (visible row 19).
class Board {
public:
    Board() { clear(); }

    void clear() {
        std::memset(rows_, 0, sizeof(rows_));
        heights_cache_valid_ = false;
    }

    // Get raw row data.
    uint16_t row(int r) const { return rows_[r]; }
    const uint16_t* data() const { return rows_; }

    // Set a cell.
    void set(int col, int row, bool occupied) {
        if (occupied)
            rows_[row] |= (1u << col);
        else
            rows_[row] &= ~(1u << col);
        heights_cache_valid_ = false;
    }

    // Check if a cell is occupied.
    bool get(int col, int row) const {
        return (rows_[row] >> col) & 1u;
    }

    // Place a piece's cells onto the board.
    void placePiece(const std::array<std::pair<int8_t, int8_t>, 4>& cells,
                    int origin_col, int origin_row, PieceName name) {
        for (const auto& [cx, cy] : cells) {
            int col = origin_col + cx;
            int row = origin_row + cy;
            if (row >= 0 && row < TOTAL_ROWS && col >= 0 && col < COLS) {
                set(col, row, true);
            }
        }
        heights_cache_valid_ = false;
    }

    // Check collision for a piece's cells at given origin position.
    bool collides(const std::array<std::pair<int8_t, int8_t>, 4>& cells,
                  int origin_col, int origin_row) const {
        for (const auto& [cx, cy] : cells) {
            int col = origin_col + cx;
            int row = origin_row + cy;
            if (col < 0 || col >= COLS) return true;
            if (row >= TOTAL_ROWS) return true;
            if (row < 0) continue;  // above board is OK
            if (get(col, row)) return true;
        }
        return false;
    }

    // Check if piece at origin is valid (no collision, within bounds).
    bool isValid(const std::array<std::pair<int8_t, int8_t>, 4>& cells,
                 int origin_col, int origin_row) const {
        return !collides(cells, origin_col, origin_row);
    }

    // Clear completed lines. Returns number of lines cleared.
    int clearLines() {
        int cleared = 0;
        int write_row = TOTAL_ROWS - 1;

        for (int read_row = TOTAL_ROWS - 1; read_row >= 0; read_row--) {
            if (rows_[read_row] == FULL_ROW) {
                cleared++;
            } else {
                if (write_row != read_row) {
                    rows_[write_row] = rows_[read_row];
                }
                write_row--;
            }
        }
        // Fill top rows with zeros.
        for (int r = write_row; r >= 0; r--) {
            rows_[r] = 0;
        }

        heights_cache_valid_ = false;
        return cleared;
    }

    // Compute ghost Y (drop distance) for a piece.
    int ghostY(const std::array<std::pair<int8_t, int8_t>, 4>& cells,
               int origin_col, int origin_row) const {
        int y = origin_row;
        while (true) {
            if (collides(cells, origin_col, y + 1)) break;
            y++;
        }
        return y;
    }

    // Column heights (0-indexed from visible top = HIDDEN_ROWS).
    const std::array<int, COLS>& columnHeights() const {
        if (!heights_cache_valid_) {
            computeHeights();
        }
        return heights_;
    }

    // Count holes (empty cells under a filled cell in same column).
    int countHoles() const {
        int holes = 0;
        for (int col = 0; col < COLS; col++) {
            bool found_block = false;
            for (int row = 0; row < TOTAL_ROWS; row++) {
                if (get(col, row)) {
                    found_block = true;
                } else if (found_block) {
                    holes++;
                }
            }
        }
        return holes;
    }

    // Compute bumpiness = sum of absolute height differences between adjacent columns.
    int bumpiness() const {
        const auto& h = columnHeights();
        int bump = 0;
        for (int c = 0; c < COLS - 1; c++) {
            bump += std::abs(h[c] - h[c + 1]);
        }
        return bump;
    }

    // Max well depth.
    int maxWellDepth() const {
        const auto& h = columnHeights();
        int max_well = 0;
        for (int c = 0; c < COLS; c++) {
            int left_diff = (c > 0) ? h[c - 1] - h[c] : 0;
            int right_diff = (c < COLS - 1) ? h[c + 1] - h[c] : 0;
            int well = std::max(0, std::min(left_diff, right_diff));
            if (well > max_well) max_well = well;
        }
        return max_well;
    }

    // Total column height sum.
    int totalHeight() const {
        const auto& h = columnHeights();
        int total = 0;
        for (int c = 0; c < COLS; c++) total += h[c];
        return total;
    }

    // Row transitions (number of horizontal cell-type changes per row).
    int rowTransitions() const {
        int transitions = 0;
        for (int r = 0; r < TOTAL_ROWS; r++) {
            for (int c = 0; c <= COLS; c++) {  // c==COLS checks boundary
                bool left = (c > 0) && get(c - 1, r);
                bool right = (c < COLS) && get(c, r);
                if (left != right) transitions++;
            }
        }
        return transitions;
    }

    // Check if any blocks occupy hidden rows (game over condition).
    bool blocksInHiddenRows() const {
        for (int r = 0; r < HIDDEN_ROWS; r++) {
            if (rows_[r] != 0) return true;
        }
        return false;
    }

private:
    void computeHeights() const {
        for (int col = 0; col < COLS; col++) {
            int h = TOTAL_ROWS;
            for (int row = 0; row < TOTAL_ROWS; row++) {
                if (get(col, row)) {
                    h = row;
                    break;
                }
            }
            heights_[col] = TOTAL_ROWS - h;
        }
        heights_cache_valid_ = true;
    }

    uint16_t rows_[TOTAL_ROWS];
    mutable std::array<int, COLS> heights_;
    mutable bool heights_cache_valid_ = false;
};

} // namespace tetris
