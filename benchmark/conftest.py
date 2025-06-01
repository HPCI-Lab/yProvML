import psutil
import platform
import zarr
import netCDF4
import numcodecs

def pytest_report_header(config):
    mem = psutil.virtual_memory()
    return f"ram -- total: {round(mem.total / (1024 ** 3), 2):.2f} GB, available: {round(mem.available / (1024 ** 3), 2):.2f} GB\nzarr-{zarr.__version__}, netCDF4-{netCDF4.__version__}, numcodecs-{numcodecs.__version__}"

def pytest_benchmark_update_machine_info(config, machine_info):
    # Ram info
    mem = psutil.virtual_memory()
    machine_info['ram_total_gb'] = round(mem.total / (1024 ** 3), 2)
    machine_info['ram_available_gb'] = round(mem.available / (1024 ** 3), 2)

    # Disk info
    root_path = "C:\\\\" if platform.system() == "Windows" else "/"
    usage = psutil.disk_usage(root_path)
    machine_info['disk_total_gb'] = round(usage.total / (1024 ** 3), 2)
    machine_info['disk_free_gb'] = round(usage.free / (1024 ** 3), 2)

    # Library versions
    machine_info['zarr_version'] = zarr.__version__
    machine_info['netCDF4_version'] = netCDF4.__version__
    machine_info['numcodecs_version'] = numcodecs.__version__
