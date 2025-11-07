{ pkgs, ... }:
{
  packages = with pkgs; [
    rustToolchain
    cargo-deny
    cargo-edit
    cargo-watch
    rust-analyzer
    pkg-config
    openssl
  ];
  env = {
    RUST_SRC_PATH = "${pkgs.rustToolchain}/lib/rustlib/src/rust/library";
  };

  path = [
    "/home/mosheng/.cargo"
  ];

  overlay = final: prev: {
    rustToolchain =
      let
        rust = prev.rust-bin;
      in
      if builtins.pathExists ./rust-toolchain.toml then
        rust.fromRustupToolchainFile ./rust-toolchain.toml
      else if builtins.pathExists ./rust-toolchain then
        rust.fromRustupToolchainFile ./rust-toolchain
      else
        rust.stable.latest.default.override {
          extensions = [
            "rust-src"
            "rustfmt"
          ];
        };
  };
}
