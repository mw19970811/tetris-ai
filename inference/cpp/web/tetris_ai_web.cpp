// WebAssembly entry point for ONNX Runtime Web + Emscripten.
// Exposes a C API that JavaScript can call via Emscripten's cwrap.

#include <emscripten.h>
#include <string>
#include <cstring>
#include "../model_loader.h"
#include "../player.h"

namespace {

tetris::inference::AIPlayer* g_player = nullptr;
tetris::TetrisEnv* g_env = nullptr;
tetris::EnvConfig g_config;

} // anonymous namespace

extern "C" {

EMSCRIPTEN_KEEPALIVE
int init_ai(const char* model_path) {
    try {
        g_player = new tetris::inference::AIPlayer(model_path, false);
        g_env = new tetris::TetrisEnv(g_config);
        g_env->reset();
        return 0;  // success
    } catch (...) {
        return -1;
    }
}

EMSCRIPTEN_KEEPALIVE
void destroy_ai() {
    delete g_player; g_player = nullptr;
    delete g_env; g_env = nullptr;
}

EMSCRIPTEN_KEEPALIVE
int select_action() {
    if (!g_player || !g_env) return 0;
    auto action = g_player->selectAction(*g_env);
    // Pack rotation(2) + column(5) + hold(1) into one int.
    int packed = (action.rotation & 0x3)
               | ((action.column + 2) & 0x1F) << 2
               | (action.hold_first ? 1 : 0) << 7;
    return packed;
}

EMSCRIPTEN_KEEPALIVE
void step_env(int action_packed, float* reward, int* done, int* score, int* lines) {
    if (!g_env) return;

    tetris::Action action;
    action.rotation = action_packed & 0x3;
    action.column = ((action_packed >> 2) & 0x1F) - 2;
    action.hold_first = (action_packed >> 7) & 1;

    float r = g_env->step(action);
    *reward = r;
    *done = g_env->isTerminated() ? 1 : 0;
    *score = g_env->getScore();
    *lines = g_env->getLinesCleared();
}

EMSCRIPTEN_KEEPALIVE
void reset_env(int seed) {
    if (g_env) {
        g_env->reset();
    }
}

EMSCRIPTEN_KEEPALIVE
const float* get_board_data() {
    static float data[TOTAL_ROWS * COLS];
    if (!g_env) return data;
    const auto& board = g_env->getBoard();
    for (int r = 0; r < TOTAL_ROWS; r++)
        for (int c = 0; c < COLS; c++)
            data[r * COLS + c] = board.get(c, r) ? 1.0f : 0.0f;
    return data;
}

} // extern "C"
