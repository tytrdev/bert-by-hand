#pragma once

// This is all manual for now
// Probably regret it later, but better than python codegen
namespace model {

constexpr int VOCAB_SIZE = 30522;
constexpr int HIDDEN = 384;
constexpr int NUM_LAYERS = 6;
constexpr int NUM_HEADS = 12;
constexpr int HEAD_DIM = 32;
constexpr int FFN_DIM = 1536;
constexpr int MAX_POSITION = 512;
constexpr int TYPE_VOCAB_SIZE = 2;
constexpr float LAYER_NORM_EPS = 1e-12f;
constexpr int BATCH = 1;
constexpr int SEQ_LEN = 128;

} // namespace model
