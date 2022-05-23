let 
   pkgs =
   import (builtins.fetchGit {
     name = "nixos-unstable-2022-05-06";
     url = "https://github.com/nixos/nixpkgs/";
     rev = "4b2827e6a180274a4df35f0754cc6353ca853998";
   }) {};
in
  pkgs.stdenv.mkDerivation {
     name = "cuda-env-shell";
     nativeBuildInputs = with pkgs; [ 
       #fhs
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
      xorg.libXext xorg.libX11 xorg.libXv xorg.libxcb xorg.libXrandr zlib 
      # nsight crash reporter needs this need this
      libkrb5
      # nsight need qt
      # yes this is way too much but I

      ncurses5
      stdenv.cc
      binutils
      gdb
     ];
     shellHook = ''
       export CUDA_PATH=${pkgs.cudatoolkit}
       # export LD_LIBRARY_PATH=${pkgs.linuxPackages.nvidia_x11}/lib
       export EXTRA_LDFLAGS="-L/lib -L${pkgs.linuxPackages.nvidia_x11}/lib"
       export EXTRA_CCFLAGS="-I/usr/include"
     #exec cuda-env
     '';
  }

