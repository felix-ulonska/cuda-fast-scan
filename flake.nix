{
  inputs.flake-utils.url = "github:numtide/flake-utils";
  inputs.nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem
      (system:
        let pkgs = (import nixpkgs { inherit system; config = { allowUnfree = true; }; });
        in
        let buildInputs = with pkgs; [
          rustc
          cargo
          gcc
          pkgconfig
          libconfig
          cmake
          rustfmt
          clippy
          clang-tools
          gdb
          cudaPackages.cudatoolkit
          linuxPackages.nvidia_x11
        ];
        rust-toolchain = pkgs.symlinkJoin {
          name = "rust-toolchain";
          paths = [pkgs.rustc pkgs.cargo pkgs.rustPlatform.rustcSrc];
        };
        fhs = pkgs.buildFHSUserEnv {
          name = "cuda-env";
          targetPkgs = pkgs: with pkgs; [ 
            git
            gitRepo
            gnupg
            autoconf
            curl
            procps
            gnumake
            utillinux
            m4
            gperf
            unzip
            cudatoolkit
            linuxPackages.nvidia_x11
            libGLU libGL
            xorg.libXi xorg.libXmu freeglut
            xorg.libXext xorg.libX11 xorg.libXv xorg.libXrandr zlib 
            ncurses5
            stdenv.cc
            gdbgui
            binutils
            rust-toolchain
          ] ;
          multiPkgs = pkgs: with pkgs; [ zlib ];
          runScript = "bash";
          profile = ''
             export CUDA_PATH=${pkgs.cudatoolkit}
             # export LD_LIBRARY_PATH=${pkgs.linuxPackages.nvidia_x11}/lib
             export EXTRA_LDFLAGS="-L/lib -L${pkgs.linuxPackages.nvidia_x11}/lib"
             export EXTRA_CCFLAGS="-I/usr/include"
           '';
         };
        in
        {
          nixpkgs.config.allowUnfree = true;
          devShells.onlyRust = pkgs.mkShell
            {
              buildInputs = buildInputs;
              RUST_SRC_PATH = "${pkgs.rust.packages.stable.rustPlatform.rustLibSrc}";
              shellHook = ''
                export CUDA_PATH=${pkgs.cudatoolkit}
              '';
            };
          devShell = pkgs.stdenv.mkDerivation {
            name = "cuda-env-shell";
            nativeBuildInputs = [
              fhs
              buildInputs
              (pkgs.python3.withPackages (ps: with ps; with pkgs.python3Packages; [
                jupyter
                ipython
                pandas
                numpy
                matplotlib
              ]))
            ];
            shellHook = "exec cuda-env";
          };
        });
}
