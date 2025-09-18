with import <nixpkgs> { };

stdenv.mkDerivation {
  name = "drokpa";
  buildInputs = [
    screen
    python310
    poetry
  ];
  shellHook = ''
    poetry shell
  '';
  LD_LIBRARY_PATH = lib.makeLibraryPath [ pkgs.stdenv.cc.cc ];
}
