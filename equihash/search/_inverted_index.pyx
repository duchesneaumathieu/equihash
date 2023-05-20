#cython: wraparound=False, nonecheck=False, cdivision=True
from libc.stdlib cimport malloc, free

cdef unsigned long FNV_prime = 0x100000001b3
cpdef unsigned long FNV1(unsigned char [:] bytes):
    cdef unsigned long hash = 0xcbf29ce484222325
    for i in range(len(bytes)):
        hash *= FNV_prime
        hash ^= bytes[i]
    return hash

cdef unsigned int MYNULL = 0xFFFFFFFF
def cython_quadratic_build(unsigned int [:] heads, unsigned int [:] links, unsigned char [:,:] codes):
    #runs in O(n**2) in worst case (when many collisions) otherwiser in O(n)
    cdef unsigned int h = 0
    cdef unsigned int i = 0
    for i in range(len(codes)):
        h = FNV1(codes[i])%len(heads)
        if heads[h] == MYNULL:
            heads[h] = i
        else:
            h = heads[h]
            while links[h] != MYNULL:
                h = links[h]
            links[h] = i
            
def cython_linear_build(unsigned int [:] heads, unsigned int [:] links, unsigned char [:,:] codes):
    #runs in O(n) but requires extra memory (nb_head * sizeof(unsigned int)).
    cdef unsigned int* tails = <unsigned int*> malloc(len(heads) * sizeof(unsigned int))
    cdef unsigned int h = 0
    cdef unsigned int j = 0
    cdef unsigned int i = 0
    for i in range(len(codes)):
        h = FNV1(codes[i])%len(heads)
        if heads[h] == MYNULL:
            heads[h] = i
        else:
            j = tails[h]
            links[j] = i
        tails[h] = i
    free(tails)
            
cdef unsigned char memoryview_equal(unsigned char [:] a, unsigned char [:] b):
    #assumes len(a)==len(b)
    cdef unsigned int i = 0
    for i in range(len(a)):
        if a[i] != b[i]:
            return 0
    return 1
            
def cython_search(unsigned char [:] code, unsigned int [:] heads, unsigned int [:] links, unsigned char [:,:] codes):
    cdef unsigned int h = heads[FNV1(code)%len(heads)]
    cdef list index = []
    while h != MYNULL:
        if memoryview_equal(code, codes[h]):
            index.append(h)
        h = links[h]
    return index