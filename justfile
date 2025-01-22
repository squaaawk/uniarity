default:
  just --list

test:
  cargo nextest run
  cargo test --doc

coverage:
  RUSTFLAGS="-C instrument-coverage" LLVM_PROFILE_FILE='cargo-test-%p-%m.profraw' cargo nextest run --target-dir target/coverage --no-fail-fast

  mkdir -p target/coverage-report/
  grcov . --binary-path ./target/coverage/debug/deps/ -s . -t html --branch -o target/coverage-report/

  rm cargo-test-*-*.profraw
  # xdg-open target/coverage-report/html/index.html
