# import ctypes
# import numpy
# E = ctypes.cdll.LoadLibrary("kernel1.so")

# l = ['bah', '0']

# LP_c_char = ctypes.POINTER(ctypes.c_char)
# LP_LP_c_char = ctypes.POINTER(LP_c_char)

# E.main.argtypes = (ctypes.c_int, # argc
#                         LP_LP_c_char) # argv

# argc = len(l)
# argv = (LP_c_char * (argc + 1))()
# for i, arg in enumerate(l):
#     enc_arg = arg.encode('utf-8')
#     argv[i] = ctypes.create_string_buffer(enc_arg)

# E.main(argc, argv)



# # print res
# # v = ctypes.c_void_p(numpy.array([0], dtype=numpy.float32))
# # print(v)
# # E.main(ctypes.c_int(2), v)

from ctypes import cdll
from ctypes import util

so_name = util.find_library('cudart')
print('****' + so_name)
try:
    cdll.LoadLibrary('libcudart.so')
except:
    print('fail')

