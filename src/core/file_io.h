#pragma once

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

template <typename T>
std::vector<T> read_bin(const std::string &path, size_t expected_count) {
  std::FILE *f = std::fopen(path.c_str(), "rb");

  if (!f) {
    std::fprintf(stderr, "read_bin: cannot open %s\n", path.c_str());
    std::abort();
  }

  std::vector<T> out(expected_count);
  size_t got = std::fread(out.data(), sizeof(T), expected_count, f);
  std::fclose(f);

  if (got != expected_count) {
    std::fprintf(stderr, "read_bin: %s: expected %zu elems, got %zu\n",
                 path.c_str(), expected_count, got);
    std::abort();
  }

  return out;
}
