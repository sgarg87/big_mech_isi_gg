import numpy as np

is_kernel_dtype_lower_precision = True
if is_kernel_dtype_lower_precision:
    kernel_dtype_np = np.float16
else:
    kernel_dtype_np = None

