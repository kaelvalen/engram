{
  description = "PRISM — modality-portable hybrid linear-recurrent (SSD + Gated Delta) backbone";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
        };

        # CUDA toolkit for nvcc and compilation headers.  We intentionally do
        # NOT ship PyTorch from nixpkgs here: the project pins its Python deps
        # via pyproject.toml and installs them with uv inside the dev venv.
        # This avoids duplicating a huge nixpkgs CUDA PyTorch build with a
        # PyPI wheel that ends up overriding it anyway, and makes it practical
        # to track fast-moving packages such as flash-linear-attention.
        cudaToolkit = pkgs.cudaPackages.cudatoolkit;
      in
      {
        devShells.default = pkgs.mkShell {
          name = "prism-dev";

          buildInputs = with pkgs; [
            uv
            cudaToolkit
            git
            just
          ];

          env = {
            CUDA_PATH = "${cudaToolkit}";
            CUDA_HOME = "${cudaToolkit}";
            CUDA_ROOT = "${cudaToolkit}";
            PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True";
          };

          shellHook = ''
            export PATH="${cudaToolkit}/bin:$PATH"

            # Use the running system's NVIDIA driver libs when on NixOS,
            # otherwise fall back to the Nix-provided CUDA toolkit libs.
            # This avoids hard-coding pkgs.linuxPackages.nvidia_x11, whose
            # version may not match the currently loaded kernel driver
            # (critical for Blackwell/sm_120 and similar new hardware).
            if [ -d /run/opengl-driver/lib ]; then
              export LD_LIBRARY_PATH="/run/opengl-driver/lib:/run/opengl-driver-32/lib:${cudaToolkit}/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            else
              export LD_LIBRARY_PATH="${cudaToolkit}/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            fi

            VENV_DIR="$PWD/.venv"
            PYTHON_BIN="${pkgs.python312}/bin/python3.12"

            if [ ! -d "$VENV_DIR" ]; then
              echo "Creating uv virtual environment in $VENV_DIR ..."
              uv venv --python "$PYTHON_BIN" "$VENV_DIR"
            fi

            source "$VENV_DIR/bin/activate"

            # Keep the dev environment in sync with pyproject.toml.  uv is
            # incremental, so repeated shell entries are cheap.
            echo "Syncing PRISM dependencies ..."
            uv pip install -e ".[train,test,dev]"

            # flash-linear-attention is an optional production backend.  Its
            # Triton kernels are sensitive to driver/toolkit versions, so a
            # failure here is reported but does not block the shell.
            if ! uv pip install "flash-linear-attention>=0.3.2,<0.4"; then
              echo ""
              echo "WARNING: flash-linear-attention could not be installed."
              echo "         PRISM will fall back to pure-PyTorch reference paths."
            fi

            echo ""
            echo "PRISM dev shell ready."
            echo "  Python: $(python --version)"
            echo "  uv: $(uv --version)"
            if command -v nvcc >/dev/null 2>&1; then
              echo "  CUDA: $(nvcc --version | sed -n '2p' | xargs)"
            fi
            python -c "import torch; print(f'  PyTorch: {torch.__version__}  CUDA available: {torch.cuda.is_available()}')" || true
          '';
        };
      });
}
