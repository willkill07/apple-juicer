# Apple Juicer

A way to squeeze the compute and/or memory performance out of your Apple product.

## Benchmarks
- GEMM: compute-bound workload evaluating the compute performance of the AMX subsystem
- STREAM: memory-bound workload evaluating the memory performance of the M-series memory subsystem

## Prerequisites

### Hardware

- Apple M-series processor

### Software

I've intentionally kept the prerequisites minimal to potentially enable more folks to be able to easily run these benchmarks.

- Operating System: macOS Sequoia. Older versions may work, just not tested.
- AppleClang installed (default C++ compiler as part of Developer Tools on macOS)
  - Another compiler may work; however, prelimary C++23 support is required for `std::print`
- Homebrew with `libomp` package installed to the _default_ location (`/opt/homebrew`)

## Running

1. Clone the repository:

```bash
git clone https://www.github.com/willkill07/apple-juicer.git
```

2. Switch to the repository directory:

```bash
cd apple-juicer
```

3. Build the executable files:

```bash
make
```

4. Invoke your executable of choice (either `./gemm` or `./stream`)

```bash
./stream
./gemm
```

## Questions / Issues / Comments

Feel free to open any issues or PRs. Contributions are welcome.
