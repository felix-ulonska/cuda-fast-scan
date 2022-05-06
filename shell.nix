let 
   pkgs =
   import (builtins.fetchGit {
     # Descriptive name to make the store path easier to identify
     name = "nixos-unstable-2022-05-06";
     url = "https://github.com/nixos/nixpkgs/";
     # Commit hash for nixos-unstable as of 2018-09-12
     # `git ls-remote https://github.com/nixos/nixpkgs nixos-unstable`
     ref = "refs/heads/nixos-unstable";
     #rev = "c777cdf5c564015d5f63b09cc93bef4178b19b01";
   }) {};
in 
pkgs.stdenv.mkDerivation {
   name = "cuda-env-shell";
   buildInputs = with pkgs; [
     git gitRepo gnupg autoconf curl
     procps gnumake utillinux m4 gperf unzip
     cudatoolkit linuxPackages.nvidia_x11
     libGLU libGL
     xorg.libXi xorg.libXmu freeglut
     xorg.libXext xorg.libX11 xorg.libXv xorg.libXrandr zlib 
     ncurses5 stdenv.cc binutils
   ];
   shellHook = ''
      export CUDA_PATH=${pkgs.cudatoolkit}
      # export LD_LIBRARY_PATH=${pkgs.linuxPackages.nvidia_x11}/lib:${pkgs.ncurses5}/lib
      export EXTRA_LDFLAGS="-L/lib -L${pkgs.linuxPackages.nvidia_x11}/lib"
      export EXTRA_CCFLAGS="-I/usr/include"
   '';          
}
