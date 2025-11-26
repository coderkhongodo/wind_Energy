#!/usr/bin/env python
# coding: utf-8
"""
GPU Configuration Helper
Cấu hình GPU cho TensorFlow/Keras training
"""

import tensorflow as tf
import os


def configure_gpu(memory_growth=True, gpu_id=None):
    """
    Cấu hình GPU cho TensorFlow
    
    Parameters:
    -----------
    memory_growth : bool
        Nếu True, chỉ allocate GPU memory khi cần (tránh allocate toàn bộ)
    gpu_id : int or None
        Chỉ định GPU cụ thể (0, 1, ...). None = sử dụng tất cả GPU
    
    Returns:
    --------
    dict : Thông tin về GPU configuration
    """
    gpus = tf.config.list_physical_devices('GPU')
    
    if not gpus:
        print("⚠️  Không phát hiện GPU, sẽ sử dụng CPU")
        return {
            'gpu_available': False,
            'gpu_count': 0,
            'gpu_names': []
        }
    
    try:
        if gpu_id is not None:
            # Chỉ sử dụng GPU cụ thể
            if gpu_id < len(gpus):
                tf.config.set_visible_devices(gpus[gpu_id], 'GPU')
                gpus = [gpus[gpu_id]]
            else:
                print(f"⚠️  GPU {gpu_id} không tồn tại, sử dụng GPU đầu tiên")
        
        # Cấu hình memory growth
        if memory_growth:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        
        # Log thông tin GPU
        gpu_names = [gpu.name for gpu in gpus]
        print(f"✅ GPU được phát hiện: {len(gpus)} GPU(s)")
        for i, gpu in enumerate(gpus):
            print(f"   GPU {i}: {gpu.name}")
            if hasattr(gpu, 'device_details'):
                print(f"      Details: {gpu.device_details}")
        
        return {
            'gpu_available': True,
            'gpu_count': len(gpus),
            'gpu_names': gpu_names
        }
        
    except RuntimeError as e:
        print(f"❌ Lỗi cấu hình GPU: {e}")
        return {
            'gpu_available': False,
            'gpu_count': 0,
            'gpu_names': [],
            'error': str(e)
        }


def disable_gpu():
    """
    Vô hiệu hóa GPU, chỉ sử dụng CPU
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    print("🔧 GPU đã được vô hiệu hóa, sử dụng CPU")


def enable_gpu(gpu_id=None):
    """
    Kích hoạt GPU
    
    Parameters:
    -----------
    gpu_id : int or None
        Chỉ định GPU cụ thể. None = sử dụng tất cả
    """
    if gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        print(f"🔧 Chỉ sử dụng GPU {gpu_id}")
    else:
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            del os.environ['CUDA_VISIBLE_DEVICES']
        print("🔧 Sử dụng tất cả GPU có sẵn")


def get_gpu_info():
    """
    Lấy thông tin chi tiết về GPU
    
    Returns:
    --------
    dict : Thông tin GPU
    """
    gpus = tf.config.list_physical_devices('GPU')
    
    if not gpus:
        return {
            'gpu_available': False,
            'gpu_count': 0
        }
    
    info = {
        'gpu_available': True,
        'gpu_count': len(gpus),
        'gpus': []
    }
    
    for i, gpu in enumerate(gpus):
        gpu_info = {
            'id': i,
            'name': gpu.name
        }
        
        # Thử lấy thêm thông tin nếu có
        try:
            details = tf.config.experimental.get_device_details(gpu)
            if details:
                gpu_info['details'] = details
        except:
            pass
        
        info['gpus'].append(gpu_info)
    
    return info


def print_gpu_summary():
    """
    In tóm tắt thông tin GPU
    """
    print("=" * 50)
    print("GPU CONFIGURATION SUMMARY")
    print("=" * 50)
    
    info = get_gpu_info()
    
    if info['gpu_available']:
        print(f"✅ {info['gpu_count']} GPU(s) available:")
        for gpu in info['gpus']:
            print(f"   - {gpu['name']}")
    else:
        print("⚠️  No GPU available, using CPU")
    
    print("=" * 50)


# Auto-configure khi import (optional)
if __name__ == "__main__":
    # Test GPU configuration
    print_gpu_summary()
    config = configure_gpu()
    print(f"\nConfiguration result: {config}")

