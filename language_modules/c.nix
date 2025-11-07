{ pkgs, ... }:
{
  packages = with pkgs; [
    clang-tools
    cmake
    codespell
    conan
    cppcheck
    doxygen
    gtest
    lcov
    vcpkg
    vcpkg-tool
    gdb
  ];
  env = { };
  path = [ ];
  overlay = final: prev: { };

}
