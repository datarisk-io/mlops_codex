{ pkgs ? import <nixpkgs> {} }:
(pkgs.buildFHSUserEnv {
  name = "pipzone";
  targetPkgs = pkgs: (with pkgs; [
    python38
    python38Packages.pip
    python38Packages.poetry
    python38Packages.virtualenv
    zlib
    gcc
  ]);
  runScript = "bash";
}).env