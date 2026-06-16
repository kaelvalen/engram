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

        python = pkgs.python312;

        # Nix-packaged Python dependencies.  PyTorch is built with CUDA support
        # because we enabled config.cudaSupport above.
        pythonDeps = ps: with ps; [
          torch
          torchvision
          torchaudio
          tensorboard
          pyyaml
          pandas
          pytest
          hypothesis
          scikit-learn
          ruff
        ];

        pythonEnv = python.withPackages pythonDeps;

        # CUDA toolkit used for nvcc and lib paths.  Keep it in sync with the
        # PyTorch CUDA major version when possible.
        cudaToolkit = pkgs.cudaPackages.cudatoolkit;
      in
      {
        packages.default = python.pkgs.buildPythonApplication {
          pname = "prism";
          version = "0.1.0";
          pyproject = true;
          src = ./.;
          build-system = [ python.pkgs.setuptools ];
          propagatedBuildInputs = (pythonDeps python.pkgs) ++ [
            python.pkgs.setuptools
          ];
          meta = {
            description = "PRISM — modality-portable hybrid linear-recurrent backbone";
            license = pkgs.lib.licenses.mit;
          };
        };

        devShells.default = pkgs.mkShell {
          name = "prism-dev";

          buildInputs = with pkgs; [
            pythonEnv
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
            export LD_LIBRARY_PATH="${cudaToolkit}/lib:${pkgs.linuxPackages.nvidia_x11}/lib:$LD_LIBRARY_PATH"
            export EXTRA_LDFLAGS="-L/lib -L${cudaToolkit}/lib"
            export EXTRA_CCFLAGS="-I/usr/include"

            VENV_DIR="$PWD/.venv"
            if [ ! -d "$VENV_DIR" ]; then
              echo "Creating Python virtual environment in $VENV_DIR ..."
              ${python}/bin/python -m venv "$VENV_DIR"
            fi
            source "$VENV_DIR/bin/activate"

            # Ensure the project itself is installed in editable mode, including
            # the optional GPU kernels if they are available on the current platform.
            if ! python -c "import prism" 2>/dev/null; then
              echo "Installing PRISM in editable mode ..."
              pip install -e ".[train,test,dev]"
              # GPU kernels are optional; FLA is the most important production backend.
              pip install "flash-linear-attention>=0.3.2,<0.4" || true
            fi

            echo ""
            echo "PRISM dev shell ready."
            echo "  Python: $(python --version)"
            echo "  CUDA:   $(nvcc --version | sed -n '2p' | xargs)"
            python -c "import torch; print(f'  PyTorch: {torch.__version__}  CUDA available: {torch.cuda.is_available()}')"
          '';
        };
      });
}
