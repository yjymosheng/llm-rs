{ pkgs, ... }:
{
  packages = with pkgs; [
    node2nix
    nodejs
    nodePackages.pnpm
    yarn
  ];
  env = { };
  path = [ ];
  overlay = final: prev: { };
}
