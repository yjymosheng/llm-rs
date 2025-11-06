{ pkgs, ... }:
{
  packages = with pkgs; [
    cabal-install
    ghc
    haskell-language-server
  ];
  env = { };
  path = [ ];
  overlay = final: prev: { };
}
