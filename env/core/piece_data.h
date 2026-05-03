#pragma once
#include <cstdint>
#include <array>
#include <string>
#include <unordered_map>
#include <vector>

namespace tetris {

constexpr int COLS = 10;
constexpr int ROWS = 20;
constexpr int HIDDEN_ROWS = 2;
constexpr int TOTAL_ROWS = ROWS + HIDDEN_ROWS;
constexpr uint16_t FULL_ROW = (1 << COLS) - 1;  // 0x3FF

enum class PieceName : uint8_t { I = 0, O = 1, T = 2, S = 3, Z = 4, J = 5, L = 6, NONE = 7 };
constexpr const char* PIECE_NAMES_STR[] = {"I", "O", "T", "S", "Z", "J", "L"};

// Each piece has 4 rotation states, each rotation is an array of 4 (x,y) cell offsets.
// All pieces are stored with rotation states 0,1,2,3 (clockwise rotations).
// Each cell is {col_offset, row_offset} relative to the piece origin.
struct PieceDef {
    std::array<std::array<std::pair<int8_t, int8_t>, 4>, 4> rotations;
};

const PieceDef PIECES[] = {
    // I
    std::array<std::array<std::pair<int8_t, int8_t>, 4>, 4>
    {
        std::array<std::pair<int8_t, int8_t>, 4>
        {
            std::pair<int8_t, int8_t>(int8_t(0), int8_t(0)),
            std::pair<int8_t, int8_t>(int8_t(1), int8_t(0)),
            std::pair<int8_t, int8_t>(int8_t(2), int8_t(0)),
            std::pair<int8_t, int8_t>(int8_t(3), int8_t(0))
        },
        {
            std::pair<int8_t, int8_t>(int8_t(0), int8_t(0)),
            std::pair<int8_t, int8_t>(int8_t(0), int8_t(1)),
            std::pair<int8_t, int8_t>(int8_t(0), int8_t(2)),
            std::pair<int8_t, int8_t>(int8_t(0), int8_t(3))
        },
        {
            std::pair<int8_t, int8_t>(int8_t(0), int8_t(0)),
            std::pair<int8_t, int8_t>(int8_t(1), int8_t(0)),
            std::pair<int8_t, int8_t>(int8_t(2), int8_t(0)),
            std::pair<int8_t, int8_t>(int8_t(3), int8_t(0))
        },
        {
            std::pair<int8_t, int8_t>(int8_t(0), int8_t(0)),
            std::pair<int8_t, int8_t>(int8_t(0), int8_t(1)),
            std::pair<int8_t, int8_t>(int8_t(0), int8_t(2)),
            std::pair<int8_t, int8_t>(int8_t(0), int8_t(3))
        }
    },
    // O
    std::array<std::array<std::pair<int8_t, int8_t>, 4>, 4>
    {
        std::array<std::pair<int8_t, int8_t>, 4>
        {
            std::pair<int8_t, int8_t>(int8_t(0), int8_t(0)),
            std::pair<int8_t, int8_t>(int8_t(1), int8_t(0)),
            std::pair<int8_t, int8_t>(int8_t(0), int8_t(1)),
            std::pair<int8_t, int8_t>(int8_t(1), int8_t(1))
        },
        {
            std::pair<int8_t, int8_t>(int8_t(0), int8_t(0)),
            std::pair<int8_t, int8_t>(int8_t(1), int8_t(0)),
            std::pair<int8_t, int8_t>(int8_t(0), int8_t(1)),
            std::pair<int8_t, int8_t>(int8_t(1), int8_t(1))
        },
        {
            std::pair<int8_t, int8_t>(int8_t(0), int8_t(0)),
            std::pair<int8_t, int8_t>(int8_t(1), int8_t(0)),
            std::pair<int8_t, int8_t>(int8_t(0), int8_t(1)),
            std::pair<int8_t, int8_t>(int8_t(1), int8_t(1))
        },
        {
            std::pair<int8_t, int8_t>(int8_t(0), int8_t(0)),
            std::pair<int8_t, int8_t>(int8_t(1), int8_t(0)),
            std::pair<int8_t, int8_t>(int8_t(0), int8_t(1)),
            std::pair<int8_t, int8_t>(int8_t(1), int8_t(1))
        }
    },
    // T
    std::array<std::array<std::pair<int8_t, int8_t>, 4>, 4>
    {
        std::array<std::pair<int8_t, int8_t>, 4>
        {
            std::pair<int8_t, int8_t>(int8_t(0), int8_t(0)),
            std::pair<int8_t, int8_t>(int8_t(1), int8_t(0)),
            std::pair<int8_t, int8_t>(int8_t(2), int8_t(0)),
            std::pair<int8_t, int8_t>(int8_t(1), int8_t(1))
        },
        {
            std::pair<int8_t, int8_t>(int8_t(0), int8_t(0)),
            std::pair<int8_t, int8_t>(int8_t(0), int8_t(1)),
            std::pair<int8_t, int8_t>(int8_t(0), int8_t(2)),
            std::pair<int8_t, int8_t>(int8_t(1), int8_t(1))
        },
        {
            std::pair<int8_t, int8_t>(int8_t(1), int8_t(0)),
            std::pair<int8_t, int8_t>(int8_t(0), int8_t(1)),
            std::pair<int8_t, int8_t>(int8_t(1), int8_t(1)),
            std::pair<int8_t, int8_t>(int8_t(2), int8_t(1))
        },
        {
            std::pair<int8_t, int8_t>(int8_t(1), int8_t(0)),
            std::pair<int8_t, int8_t>(int8_t(1), int8_t(1)),
            std::pair<int8_t, int8_t>(int8_t(1), int8_t(2)),
            std::pair<int8_t, int8_t>(int8_t(0), int8_t(1))
        }
    },
    // S
    std::array<std::array<std::pair<int8_t, int8_t>, 4>, 4>
    {
        std::array<std::pair<int8_t, int8_t>, 4>
        {
            std::pair<int8_t, int8_t>(int8_t(1), int8_t(0)),
            std::pair<int8_t, int8_t>(int8_t(2), int8_t(0)),
            std::pair<int8_t, int8_t>(int8_t(0), int8_t(1)),
            std::pair<int8_t, int8_t>(int8_t(1), int8_t(1))
        },
        {
            std::pair<int8_t, int8_t>(int8_t(0), int8_t(0)),
            std::pair<int8_t, int8_t>(int8_t(0), int8_t(1)),
            std::pair<int8_t, int8_t>(int8_t(1), int8_t(1)),
            std::pair<int8_t, int8_t>(int8_t(1), int8_t(2))
        },
        {
            std::pair<int8_t, int8_t>(int8_t(1), int8_t(0)),
            std::pair<int8_t, int8_t>(int8_t(2), int8_t(0)),
            std::pair<int8_t, int8_t>(int8_t(0), int8_t(1)),
            std::pair<int8_t, int8_t>(int8_t(1), int8_t(1))
        },
        {
            std::pair<int8_t, int8_t>(int8_t(0), int8_t(0)),
            std::pair<int8_t, int8_t>(int8_t(0), int8_t(1)),
            std::pair<int8_t, int8_t>(int8_t(1), int8_t(1)),
            std::pair<int8_t, int8_t>(int8_t(1), int8_t(2))
        }
    },
    // Z
    std::array<std::array<std::pair<int8_t, int8_t>, 4>, 4>
    {
        std::array<std::pair<int8_t, int8_t>, 4>
        {
            std::pair<int8_t, int8_t>(int8_t(0), int8_t(0)),
            std::pair<int8_t, int8_t>(int8_t(1), int8_t(0)),
            std::pair<int8_t, int8_t>(int8_t(1), int8_t(1)),
            std::pair<int8_t, int8_t>(int8_t(2), int8_t(1))
        },
        {
            std::pair<int8_t, int8_t>(int8_t(1), int8_t(0)),
            std::pair<int8_t, int8_t>(int8_t(0), int8_t(1)),
            std::pair<int8_t, int8_t>(int8_t(1), int8_t(1)),
            std::pair<int8_t, int8_t>(int8_t(0), int8_t(2))
        },
        {
            std::pair<int8_t, int8_t>(int8_t(0), int8_t(0)),
            std::pair<int8_t, int8_t>(int8_t(1), int8_t(0)),
            std::pair<int8_t, int8_t>(int8_t(1), int8_t(1)),
            std::pair<int8_t, int8_t>(int8_t(2), int8_t(1))
        },
        {
            std::pair<int8_t, int8_t>(int8_t(1), int8_t(0)),
            std::pair<int8_t, int8_t>(int8_t(0), int8_t(1)),
            std::pair<int8_t, int8_t>(int8_t(1), int8_t(1)),
            std::pair<int8_t, int8_t>(int8_t(0), int8_t(2))
        }
    },
    // J
    std::array<std::array<std::pair<int8_t, int8_t>, 4>, 4>
    {
        std::array<std::pair<int8_t, int8_t>, 4>
        {
            std::pair<int8_t, int8_t>(int8_t(0), int8_t(0)),
            std::pair<int8_t, int8_t>(int8_t(0), int8_t(1)),
            std::pair<int8_t, int8_t>(int8_t(1), int8_t(1)),
            std::pair<int8_t, int8_t>(int8_t(2), int8_t(1))
        },
        {
            std::pair<int8_t, int8_t>(int8_t(0), int8_t(0)),
            std::pair<int8_t, int8_t>(int8_t(1), int8_t(0)),
            std::pair<int8_t, int8_t>(int8_t(0), int8_t(1)),
            std::pair<int8_t, int8_t>(int8_t(0), int8_t(2))
        },
        {
            std::pair<int8_t, int8_t>(int8_t(0), int8_t(0)),
            std::pair<int8_t, int8_t>(int8_t(1), int8_t(0)),
            std::pair<int8_t, int8_t>(int8_t(2), int8_t(0)),
            std::pair<int8_t, int8_t>(int8_t(2), int8_t(1))
        },
        {
            std::pair<int8_t, int8_t>(int8_t(1), int8_t(0)),
            std::pair<int8_t, int8_t>(int8_t(1), int8_t(1)),
            std::pair<int8_t, int8_t>(int8_t(0), int8_t(2)),
            std::pair<int8_t, int8_t>(int8_t(1), int8_t(2))
        }
    },
    // L
    std::array<std::array<std::pair<int8_t, int8_t>, 4>, 4>
    {
        std::array<std::pair<int8_t, int8_t>, 4>
        {
            std::pair<int8_t, int8_t>(int8_t(2), int8_t(0)),
            std::pair<int8_t, int8_t>(int8_t(0), int8_t(1)),
            std::pair<int8_t, int8_t>(int8_t(1), int8_t(1)),
            std::pair<int8_t, int8_t>(int8_t(2), int8_t(1))
        },
        {
            std::pair<int8_t, int8_t>(int8_t(0), int8_t(0)),
            std::pair<int8_t, int8_t>(int8_t(0), int8_t(1)),
            std::pair<int8_t, int8_t>(int8_t(0), int8_t(2)),
            std::pair<int8_t, int8_t>(int8_t(1), int8_t(2))
        },
        {
            std::pair<int8_t, int8_t>(int8_t(0), int8_t(0)),
            std::pair<int8_t, int8_t>(int8_t(1), int8_t(0)),
            std::pair<int8_t, int8_t>(int8_t(2), int8_t(0)),
            std::pair<int8_t, int8_t>(int8_t(0), int8_t(1))
        },
        {
            std::pair<int8_t, int8_t>(int8_t(0), int8_t(0)),
            std::pair<int8_t, int8_t>(int8_t(1), int8_t(0)),
            std::pair<int8_t, int8_t>(int8_t(1), int8_t(1)),
            std::pair<int8_t, int8_t>(int8_t(1), int8_t(2))
        }
    }
};

// SRS Wall Kick data: key = "fromRotation>toRotation"
// Normal pieces and I-piece have separate kick tables.
// Each kick test is {dx, dy} where dx is column offset, dy is row offset (positive = down).
struct KickEntry { int8_t dx; int8_t dy; };

inline const std::unordered_map<std::string, std::vector<KickEntry>> WALL_KICKS_NORMAL = {
    {"0>1", {{0,0},{-1,0},{-1,-1},{0,2},{-1,2}}},
    {"1>0", {{0,0},{1,0},{1,1},{0,-2},{1,-2}}},
    {"1>2", {{0,0},{1,0},{1,1},{0,-2},{1,-2}}},
    {"2>1", {{0,0},{-1,0},{-1,-1},{0,2},{-1,2}}},
    {"2>3", {{0,0},{1,0},{1,-1},{0,2},{1,2}}},
    {"3>2", {{0,0},{-1,0},{-1,1},{0,-2},{-1,-2}}},
    {"3>0", {{0,0},{-1,0},{-1,1},{0,-2},{-1,-2}}},
    {"0>3", {{0,0},{1,0},{1,-1},{0,2},{1,2}}}
};

inline const std::unordered_map<std::string, std::vector<KickEntry>> WALL_KICKS_I = {
    {"0>1", {{0,0},{-2,0},{1,0},{-2,1},{1,-2}}},
    {"1>0", {{0,0},{2,0},{-1,0},{2,-1},{-1,2}}},
    {"1>2", {{0,0},{-1,0},{2,0},{-1,-2},{2,1}}},
    {"2>1", {{0,0},{1,0},{-2,0},{1,2},{-2,-1}}},
    {"2>3", {{0,0},{2,0},{-1,0},{2,-1},{-1,2}}},
    {"3>2", {{0,0},{-2,0},{1,0},{-2,1},{1,-2}}},
    {"3>0", {{0,0},{1,0},{-2,0},{1,2},{-2,-1}}},
    {"0>3", {{0,0},{-1,0},{2,0},{-1,-2},{2,1}}}
};

// Scoring table: lines_cleared -> base_score  (amplified to incentivise clears)
inline constexpr int SCORE_TABLE[] = {0, 150, 500, 1000, 2000};

// Speed curve: drop interval in ms per level (level 1 = index 0)
inline constexpr int DROP_SPEEDS[] = {
    1000, 800, 650, 500, 400, 320, 250, 180, 130, 90,
    70, 55, 45, 35, 28, 22, 17, 14, 11, 9
};

// Lock delay parameters
constexpr int LOCK_DELAY_MS = 500;
constexpr int LOCK_MOVES_MAX = 15;

} // namespace tetris
