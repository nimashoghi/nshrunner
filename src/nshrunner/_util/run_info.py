def _dump_run_information(self, config: BaseConfig):
    try:
        import yaml

    except ImportError:
        log.warning("Failed to import `yaml`. Skipping dumping of run information.")
        return

    dump_dir = config.directory.resolve_subdirectory(config.id, "stdio") / "dump"

    # Create a different directory for each rank.
    # Easy way for now: Add a random subdir.
    dump_dir = dump_dir / f"rank_{str(uuid.uuid4())}"
    dump_dir.mkdir(parents=True, exist_ok=True)

    # First, dump the full config
    full_config_path = dump_dir / "config.yaml"
    config_dict = config.model_dump(mode="json")
    with full_config_path.open("w") as file:
        yaml.dump(config_dict, file)

    # Dump all environment variables
    env_vars_path = dump_dir / "env.yaml"
    env_vars = dict(os.environ)
    with env_vars_path.open("w") as file:
        yaml.dump(env_vars, file)

    # Dump the output of `nvidia-smi` to a file (if available)
    # First, resolve either `nvidia-smi` or `rocm-smi` (for AMD GPUs)
    if not (smi_exe := self._resolve_gpu_smi()):
        return

    nvidia_smi_path = dump_dir / "nvidia_smi_output.log"
    try:
        with nvidia_smi_path.open("w") as file:
            subprocess.run([smi_exe], stdout=file, stderr=subprocess.PIPE)
    except FileNotFoundError:
        log.warning(f"Failed to run `{smi_exe}`.")


def _resolve_gpu_smi(self):
    if shutil.which("nvidia-smi"):
        return "nvidia-smi"
    elif shutil.which("rocm-smi"):
        return "rocm-smi"
    else:
        log.warning("No GPU monitoring tool found.")
        return None
