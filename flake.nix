{
  description = "MLIR";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (
      system:
        let
          pkgs = import nixpkgs {
            inherit system;
          };
        in
          {
            devShells.default = with pkgs; mkShell {
              buildInputs = [
	            pkgs.gdb pkgs.clang-tools pkgs.cmake pkgs.ninja
		        pkgs.ccache pkgs.clang pkgs.lld
              ];
              LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib/;${pkgs.llvmPackages.openmp}/lib";
            };
          }
    );
}
