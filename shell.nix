with import <nixpkgs> { };

stdenv.mkDerivation {
  name = "drokpa";
  buildInputs = [
    screen
    python310
    poetry
    zlib
    libGL
    glib
    opencv4
    pkg-config
  ];
  shellHook = ''
    poetry shell
  '';
  LD_LIBRARY_PATH = lib.makeLibraryPath [
    pkgs.stdenv.cc.cc
    pkgs.zlib
    pkgs.libGL
    pkgs.glib
  ];
}
