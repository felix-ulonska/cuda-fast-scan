let 
   pkgs =
   import (builtins.fetchGit {
     name = "nixos-unstable-2022-05-06";
     url = "https://github.com/nixos/nixpkgs/";
     ref = "refs/heads/nixos-unstable";
   }) {};
in 
let 
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
  pkgs.stdenv.mkDerivation {
    
     name = "cuda-env-shell";
     nativeBuildInputs = with pkgs; [ 
       qt5.qtbase
       cudaPackages.nsight_systems
       fhs ];
       shellHook = ''
       # for nsight_compute https://discourse.nixos.org/t/python-qt-qpa-plugin-could-not-find-xcb/8862
       export QT_QPA_PLATFORM_PLUGIN_PATH="${pkgs.qt5.qtbase.bin}/lib/qt-${pkgs.qt5.qtbase.version}/plugins"
       exec cuda-env'';
  }

