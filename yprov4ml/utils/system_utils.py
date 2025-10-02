
import psutil
import torch
import sys
import warnings

from yprov4ml.constants import VERBOSE

if sys.platform != 'darwin':
    import GPUtil
    import gpustat

    if torch.cuda.device_count() == 0:
        if VERBOSE:
            warnings.warn("No GPU found")
    elif "AMD" in torch.cuda.get_device_name(0): 
        import pyamdgpuinfo
    elif "NVIDIA" in torch.cuda.get_device_name(0): 
        from nvitop import Device
else: 
    import apple_gpu

def get_cpu_usage() -> float:
    """
    Returns the current CPU usage percentage.
    
    Returns:
        float: The CPU usage percentage.
    """
    return psutil.cpu_percent()

def get_memory_usage() -> float:
    """
    Returns the current memory usage percentage.
    
    Returns:
        float: The memory usage percentage.
    """
    return psutil.virtual_memory().percent

def get_disk_usage() -> float:
    """
    Returns the current disk usage percentage.
    
    Returns:
        float: The disk usage percentage.
    """
    return psutil.disk_usage('/').percent

def get_gpu_memory_usage() -> float:
    """
    Returns the current GPU memory usage percentage, if GPU is available.
    
    Returns:
        float: The GPU memory usage percentage.
    """    
    if sys.platform != 'darwin':
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / torch.cuda.memory_reserved()
        else: 
            if "NVIDIA" in torch.cuda.get_device_name(0):
                devices = Device.all()
                if len(devices) == 0:  
                    return 0.0
                device = devices[0] 
                return device.memory_used / device.memory_total
            return 0.0
    else: 
        return get_gpu_metric_apple('memory')

    
def get_gpu_power_usage() -> float:
    """
    Returns the current GPU power usage percentage, if GPU is available.
    
    Returns:
        float: The GPU power usage percentage.
    """
    if sys.platform != 'darwin':
        if torch.cuda.device_count() == 0:
            return 0.0

        gpu_power = 0.0
        if torch.cuda.is_available():
            gpu_power = get_gpu_metric_gputil('power')
        if not gpu_power and "AMD" in torch.cuda.get_device_name(0):
            gpu_power = get_gpu_metric_amd('power')
    else:
        gpu_power = get_gpu_metric_apple('power')

    return gpu_power
    
def get_gpu_temperature() -> float:
    """
    Returns the current GPU temperature, if GPU is available.
    
    Returns:
        float: The GPU temperature.
    """
    if sys.platform != 'darwin':
        if torch.cuda.device_count() == 0:
            return 0.0

        gpu_temperature = 0.0
        if torch.cuda.is_available():
            gpu_temperature = get_gpu_metric_gputil('temperature')
        if not gpu_temperature and "NVIDIA" in torch.cuda.get_device_name(0): 
            gpu_utilization = get_gpu_metric_nvidia('temperature')
        if not gpu_temperature and "AMD" in torch.cuda.get_device_name(0): 
            gpu_temperature = get_gpu_metric_amd('temperature')

    else:
        gpu_temperature = get_gpu_metric_apple('temperature')

    return gpu_temperature

def get_gpu_usage() -> float:
    """
    Returns the current GPU usage percentage, if GPU is available.
    
    Returns:
        float: The GPU usage percentage.
    """
    if sys.platform != 'darwin':
        if torch.cuda.device_count() == 0:
            return 0.0
        
        gpu_utilization = 0.0
        if torch.cuda.is_available():
            gpu_utilization = get_gpu_metric_gputil('utilization')
        if not gpu_utilization and "NVIDIA" in torch.cuda.get_device_name(0):
            gpu_utilization = get_gpu_metric_nvidia('utilization')
        if not gpu_utilization and "AMD" in torch.cuda.get_device_name(0):
            gpu_utilization = get_gpu_metric_amd('utilization')
    else:
        gpu_utilization = get_gpu_metric_apple('utilization')

    return gpu_utilization

def get_gpu_metric_amd(metric): 
    try: 
        first_gpu = pyamdgpuinfo.get_gpu(0)
        if metric == 'power':
            m = first_gpu.query_power()
        elif metric == 'temperature':
            m = first_gpu.query_temperature()
        elif metric == 'utilization':
            m = first_gpu.query_utilization()

        return m
    except:
        if VERBOSE:
            warnings.warn(f"Could not get metric: {metric}")
        return 0.0

def get_gpu_metric_nvidia(metric):

    devices = Device.all()
    if len(devices) == 0:  
        return None
    device = devices[0] 

    if metric == 'temperature':
        return device.temperature()
    elif metric == "utilization": 
        return device.gpu_utilization()
    elif metric == 'fan_speed': 
        return device.fan_speed()
    elif metric == 'memory_total':
        return device.memory_total_human()
    else: 
        if VERBOSE:
            warnings.warn(f"Could not get metric: {metric}")
        return 0.0

def get_gpu_metric_gputil(metric):
    current_gpu = torch.cuda.current_device()
    gpus = GPUtil.getGPUs()
    if current_gpu < len(gpus):
        if metric == 'temperature':
            return gpus[current_gpu].temperature
        elif metric == "utilization": 
            return gpus[current_gpu].load
        else: 
            return 0.0
    else:
        if VERBOSE:
            warnings.warn(f"Could not get metric: {metric}")
        return 0.0

def get_gpu_metric_apple(metric):
    statistics = apple_gpu.accelerator_performance_statistics()
    if metric == 'power':
        return 0.0
    elif metric == 'temperature':
        return 0.0
    elif metric == 'utilization':
        return statistics['Device Utilization %']
    elif metric == 'memory':
        return statistics['Alloc system memory']
    else: 
        if VERBOSE:
            warnings.warn(f"Could not get metric: {metric}")
        return 0.0