"""
Dynamic tool loading package for Chintu.
All isolated tool scripts placed here are auto-discovered and registered at startup.
"""
import os
import importlib.util

def load_dynamic_tools() -> dict:
    tools_map = {}
    current_dir = os.path.dirname(__file__)
    for filename in os.listdir(current_dir):
        if filename.endswith(".py") and filename != "__init__.py":
            module_name = filename[:-3]
            file_path = os.path.join(current_dir, filename)
            try:
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    # Register any callable function defined in the module that doesn't start with an underscore
                    for attr_name in dir(mod):
                        attr = getattr(mod, attr_name)
                        if callable(attr) and not attr_name.startswith("_"):
                            tools_map[attr_name] = attr
            except Exception as e:
                print(f"[Dynamic Tool Load Error] Failed to load {filename}: {e}", flush=True)
    return tools_map
