#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "../core/piece_data.h"
#include "../core/randomizer.h"
#include "../core/tetris_core.h"
#include "../core/action_gen.h"
#include "../core/tetris_env.h"
#include "../core/state_encoder.h"

namespace py = pybind11;
using namespace tetris;

PYBIND11_MODULE(tetris_core, m) {
    m.doc() = "High-performance C++ Tetris environment for RL training";

    // ---- PieceName enum ----
    py::enum_<PieceName>(m, "PieceName")
        .value("I", PieceName::I)
        .value("O", PieceName::O)
        .value("T", PieceName::T)
        .value("S", PieceName::S)
        .value("Z", PieceName::Z)
        .value("J", PieceName::J)
        .value("L", PieceName::L)
        .value("NONE", PieceName::NONE)
        .export_values();

    // ---- Action ----
    py::class_<Action>(m, "Action")
        .def(py::init<>())
        .def_readwrite("rotation", &Action::rotation)
        .def_readwrite("column", &Action::column)
        .def_readwrite("hold", &Action::hold)
        .def("__repr__", [](const Action& a) {
            return "Action(rot=" + std::to_string(a.rotation)
                 + ", col=" + std::to_string(a.column)
                 + ", hold=" + (a.hold ? "True" : "False") + ")";
        });

    // ---- Board ----
    py::class_<Board>(m, "Board")
        .def(py::init<>())
        .def("clear", &Board::clear)
        .def("set", &Board::set, py::arg("col"), py::arg("row"), py::arg("occupied"))
        .def("get", &Board::get, py::arg("col"), py::arg("row"))
        .def("clear_lines", &Board::clearLines)
        .def("count_holes", &Board::countHoles)
        .def("bumpiness", &Board::bumpiness)
        .def("max_well_depth", &Board::maxWellDepth)
        .def("total_height", &Board::totalHeight)
        .def("row_transitions", &Board::rowTransitions)
        .def("blocks_in_hidden_rows", &Board::blocksInHiddenRows)
        .def("column_heights", [](const Board& b) {
            const auto& h = b.columnHeights();
            return py::array_t<int>({COLS}, h.data());
        })
        .def("to_numpy", [](const Board& b) {
            py::array_t<uint8_t> arr({TOTAL_ROWS, COLS});
            auto buf = arr.mutable_unchecked<2>();
            for (int r = 0; r < TOTAL_ROWS; r++) {
                for (int c = 0; c < COLS; c++) {
                    buf(r, c) = b.get(c, r) ? 1 : 0;
                }
            }
            return arr;
        });

    // ---- EnvConfig ----
    py::class_<EnvConfig>(m, "EnvConfig")
        .def(py::init<>())
        .def_readwrite("cols", &EnvConfig::cols)
        .def_readwrite("rows", &EnvConfig::rows)
        .def_readwrite("hidden_rows", &EnvConfig::hidden_rows)
        .def_readwrite("lock_delay_ms", &EnvConfig::lock_delay_ms)
        .def_readwrite("lock_moves_max", &EnvConfig::lock_moves_max)
        .def_readwrite("next_queue_size", &EnvConfig::next_queue_size)
        .def_readwrite("bag_type", &EnvConfig::bag_type)
        .def_readwrite("w_height", &EnvConfig::w_height)
        .def_readwrite("w_holes", &EnvConfig::w_holes)
        .def_readwrite("w_bumpiness", &EnvConfig::w_bumpiness)
        .def_readwrite("w_well", &EnvConfig::w_well)
        .def_readwrite("w_survival", &EnvConfig::w_survival)
        .def_readwrite("w_death", &EnvConfig::w_death)
        .def_readwrite("hard_drop_score", &EnvConfig::hard_drop_score)
        .def_readwrite("soft_drop_score", &EnvConfig::soft_drop_score);

    // ---- StateEncoder ----
    py::class_<StateEncoder>(m, "StateEncoder")
        .def(py::init<int, int, int, int>(),
             py::arg("cols") = COLS,
             py::arg("total_rows") = TOTAL_ROWS,
             py::arg("hidden_rows") = HIDDEN_ROWS,
             py::arg("next_queue_size") = 4)
        .def("feature_dim", &StateEncoder::featureDim)
        .def("encode_features", [](const StateEncoder& enc,
                                    const Board& board,
                                    PieceName current_piece, int current_rotation,
                                    PieceName hold_piece,
                                    const std::vector<PieceName>& next_queue) {
            auto feats = enc.encodeFeatures(board, current_piece, current_rotation,
                                            hold_piece, next_queue);
            return py::array_t<float>({StateEncoder::FEATURE_DIM}, feats.data());
        })
        .def_static("encode_board", [](const Board& board) {
            py::array_t<float> arr({1, TOTAL_ROWS, COLS});
            auto buf = arr.mutable_unchecked<3>();
            for (int r = 0; r < TOTAL_ROWS; r++) {
                uint16_t row = board.row(r);
                for (int c = 0; c < COLS; c++) {
                    buf(0, r, c) = (row >> c) & 1u ? 1.0f : 0.0f;
                }
            }
            return arr;
        });

    // ---- TetrisEnv ----
    py::class_<TetrisEnv>(m, "TetrisEnvCpp")
        .def(py::init<const EnvConfig&, uint64_t>(),
             py::arg("config") = EnvConfig(),
             py::arg("seed") = 0)
        .def("reset", &TetrisEnv::reset)
        .def("step", &TetrisEnv::step, py::arg("action"))
        .def("get_state", &TetrisEnv::getState)
        .def("get_board", &TetrisEnv::getBoard,
             py::return_value_policy::reference_internal)
        .def("get_legal_actions", &TetrisEnv::getLegalActions)
        .def("get_legal_actions_mask", [](const TetrisEnv& env, int max_actions) {
            auto mask = env.getLegalActionsMask(max_actions);
            py::array_t<bool> arr({max_actions});
            auto buf = arr.mutable_unchecked<1>();
            for (size_t i = 0; i < mask.size(); i++) {
                buf(static_cast<py::ssize_t>(i)) = mask[i];
            }
            return arr;
        }, py::arg("max_actions") = 112)
        .def("get_obs", [](TetrisEnv& env, StateEncoder& enc) {
            const auto& state = env.getState();
            // Features via encoder.
            auto feats = enc.encodeFeatures(env.getBoard(),
                state.current_piece, state.current_rotation,
                state.hold_piece, state.next_queue);
            py::array_t<float> feat_arr({StateEncoder::FEATURE_DIM});
            std::copy(feats.begin(), feats.end(), feat_arr.mutable_data());
            // Board bitmap as (1, TOTAL_ROWS, COLS) float32.
            py::array_t<float> board_arr({1, TOTAL_ROWS, COLS});
            auto buf = board_arr.mutable_unchecked<3>();
            for (int r = 0; r < TOTAL_ROWS; r++) {
                uint16_t row = state.board.row(r);
                for (int c = 0; c < COLS; c++) {
                    buf(0, r, c) = (row >> c) & 1u ? 1.0f : 0.0f;
                }
            }
            return py::make_tuple(board_arr, feat_arr);
        }, py::arg("encoder"))
        .def("get_drop_interval", &TetrisEnv::getDropInterval)
        .def("is_terminated", &TetrisEnv::isTerminated)
        .def("get_score", &TetrisEnv::getScore)
        .def("get_level", &TetrisEnv::getLevel)
        .def("get_lines_cleared", &TetrisEnv::getLinesCleared)
        .def("config", &TetrisEnv::config);

    // ---- EnvState (read-only access) ----
    py::class_<EnvState>(m, "EnvState")
        .def_readonly("current_piece", &EnvState::current_piece)
        .def_readonly("current_rotation", &EnvState::current_rotation)
        .def_readonly("hold_piece", &EnvState::hold_piece)
        .def_readonly("can_hold", &EnvState::can_hold)
        .def_readonly("next_queue", &EnvState::next_queue)
        .def_readonly("score", &EnvState::score)
        .def_readonly("level", &EnvState::level)
        .def_readonly("lines_cleared", &EnvState::lines_cleared)
        .def_readonly("terminated", &EnvState::terminated)
        .def_readonly("step_count", &EnvState::step_count);

    // ---- Constants ----
    m.attr("COLS") = COLS;
    m.attr("ROWS") = ROWS;
    m.attr("HIDDEN_ROWS") = HIDDEN_ROWS;
    m.attr("TOTAL_ROWS") = TOTAL_ROWS;
    m.attr("LOCK_DELAY_MS") = LOCK_DELAY_MS;
    m.attr("LOCK_MOVES_MAX") = LOCK_MOVES_MAX;

    // ---- Piece name list ----
    m.attr("PIECE_NAMES") = py::list(7);
    for (int i = 0; i < 7; i++) {
        m.attr("PIECE_NAMES").cast<py::list>()[i] = PIECE_NAMES_STR[i];
    }
}
