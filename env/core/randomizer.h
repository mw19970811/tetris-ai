#pragma once
#include <cstdint>
#include <array>
#include <random>
#include <vector>
#include <algorithm>
#include "piece_data.h"

namespace tetris {

// 7-bag randomizer: generates pieces in shuffled groups of 7.
// Ensures each of the 7 piece types appears exactly once per bag.
class BagRandomizer {
public:
    BagRandomizer(uint64_t seed = 0) : rng_(seed) {
        if (seed == 0) {
            std::random_device rd;
            rng_.seed(rd());
        }
    }

    void reset() {
        bag_.clear();
        bag_pos_ = 0;
    }

    void seed(uint64_t s) { rng_.seed(s); }

    // Get next piece from the bag, refilling when empty.
    PieceName next() {
        if (bag_pos_ >= bag_.size()) {
            fillBag();
            bag_pos_ = 0;
        }
        return bag_[bag_pos_++];
    }

    // Peek at the nth upcoming piece (0 = current).
    PieceName peek(size_t n = 0) const {
        // For simplicity, we regenerate if needed.
        // In production, maintain a larger buffer.
        auto temp_bag = bag_;
        auto temp_pos = bag_pos_;
        auto temp_rng = rng_;
        for (size_t i = 0; i <= n; i++) {
            if (temp_pos >= temp_bag.size()) {
                fillBagInternal(temp_bag, temp_rng);
                temp_pos = 0;
            }
            if (i == n) return temp_bag[temp_pos];
            temp_pos++;
        }
        return PieceName::I; // unreachable
    }

private:
    void fillBag() {
        fillBagInternal(bag_, rng_);
    }

    static void fillBagInternal(std::vector<PieceName>& bag, std::mt19937_64& rng) {
        bag.clear();
        for (int i = 0; i < 7; i++) {
            bag.push_back(static_cast<PieceName>(i));
        }
        std::shuffle(bag.begin(), bag.end(), rng);
    }

    mutable std::mt19937_64 rng_;
    std::vector<PieceName> bag_;
    size_t bag_pos_ = 0;
};

// Simple pure-random piece generator (for testing / alternative modes).
class RandomGenerator {
public:
    RandomGenerator(uint64_t seed = 0) : rng_(seed) {
        if (seed == 0) {
            std::random_device rd;
            rng_.seed(rd());
        }
    }

    void reset() {}
    void seed(uint64_t s) { rng_.seed(s); }

    PieceName next() {
        return static_cast<PieceName>(dist_(rng_));
    }

private:
    mutable std::mt19937_64 rng_;
    std::uniform_int_distribution<int> dist_{0, 6};
};

} // namespace tetris
