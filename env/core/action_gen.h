#pragma once
#include <cstdint>
#include <vector>
#include <string>
#include "piece_data.h"
#include "tetris_core.h"

namespace tetris {

struct Action {
    int8_t rotation;   // 0-3
    int8_t column;     // placement column (piece origin x)
    bool hold;   // true if hold should be used before placing

    bool operator==(const Action& other) const {
        return rotation == other.rotation && column == other.column && hold == other.hold;
    }
};

// Generates all legal placement actions for the current state.
// For each rotation, computes ghost position for each column where placement is valid.
// Also generates hold variants if holding is allowed.
class ActionGenerator {
public:
    // Get all legal placement actions for the current board + current piece.
    static std::vector<Action> getLegalActions(
        const Board& board,
        PieceName current_piece, int current_rotation,
        PieceName hold_piece, bool can_hold)
    {
        std::vector<Action> actions;

        // Actions without hold.
        for (int rot = 0; rot < 4; rot++) {
            const auto& cells = PIECES[static_cast<int>(current_piece)].rotations[rot];
            for (int col = -2; col < COLS + 2; col++) {
                int ghost_y = board.ghostY(cells, col, HIDDEN_ROWS - 2);
                if (ghost_y < 0) continue;
                // Check if placement at ghost_y is valid, and piece isn't spawned inside walls.
                if (board.isValid(cells, col, ghost_y)) {
                    // Verify the piece can actually reach this column (no wall collision at spawn).
                    if (!board.collides(cells, col, HIDDEN_ROWS - 2)) {
                        actions.push_back({static_cast<int8_t>(rot), static_cast<int8_t>(col), false});
                    }
                }
            }
        }

        // Actions with hold (if allowed).
        if (can_hold && hold_piece != PieceName::NONE) {
            PieceName piece_after_hold = hold_piece;
            for (int rot = 0; rot < 4; rot++) {
                const auto& cells = PIECES[static_cast<int>(piece_after_hold)].rotations[rot];
                for (int col = -2; col < COLS + 2; col++) {
                    int ghost_y = board.ghostY(cells, col, HIDDEN_ROWS - 2);
                    if (ghost_y < 0) continue;
                    if (board.isValid(cells, col, ghost_y)) {
                        if (!board.collides(cells, col, HIDDEN_ROWS - 2)) {
                            actions.push_back({static_cast<int8_t>(rot), static_cast<int8_t>(col), true});
                        }
                    }
                }
            }
        } else if (can_hold && hold_piece == PieceName::NONE) {
            // Hold with no held piece: just a flag indicating "hold and skip".
            // This is represented by a special action.
            actions.push_back({0, 0, true});  // special "hold to swap" action
        }

        return actions;
    }

    // Get drop distance (number of rows from current position to ghost).
    static int getDropDistance(const Board& board,
                                PieceName piece, int rotation,
                                int col, int current_row) {
        const auto& cells = PIECES[static_cast<int>(piece)].rotations[rotation];
        int ghost_y = board.ghostY(cells, col, current_row);
        return ghost_y - current_row;
    }
};

} // namespace tetris
