#include <cstdint>
#include <array>
#include <string>
#include <unordered_map>
#include <vector>

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

int main() {

}